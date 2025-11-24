"""
LLM Benchmark AI - Gradio Application
A tool for benchmarking GGUF models and predicting performance using ML.
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile

from engine import (
    Database, GGUFParser, BenchmarkRunner, MLPredictor,
    SmartSuggestions, HuggingFaceSearch, HardwareDetector,
    ModelMetadata, BenchmarkResult, PredictionResult,
    estimate_vram_gb
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

db = Database()
gguf_parser = GGUFParser()
benchmark_runner = BenchmarkRunner()
predictor = MLPredictor()
suggestions = SmartSuggestions(predictor)
hf_search = HuggingFaceSearch()
hardware_detector = HardwareDetector()

# Initialize hardware profile
hardware_profile = hardware_detector.detect()
db.save_hardware_profile(hardware_profile)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_confidence_bar(score: int) -> str:
    """Create visual confidence bar with color coding."""
    filled = "█" * score
    empty = "░" * (10 - score)
    if score >= 8:
        level = "HIGH"
        color = "🟢"
    elif score >= 5:
        level = "MEDIUM"
        color = "🟡"
    else:
        level = "LOW"
        color = "🔴"
    return f"{color} {filled}{empty} {score}/10 **{level}**"

def get_model_status() -> str:
    """Get current ML model status."""
    count = db.get_benchmark_count()
    if count < 2:
        return f"⏳ Need {2 - count} more benchmark(s) to train model"
    elif predictor.is_trained:
        metrics = db.get_latest_training_metrics()
        if metrics:
            return f"✅ Model trained (R²={metrics['r2']:.3f}, {metrics['num_samples']} samples)"
        return "✅ Model trained"
    else:
        return "⚠️ Model ready to train - click 'Retrain Model' in Model Performance tab"

# =============================================================================
# TAB 1: BENCHMARK RUNNER
# =============================================================================

def run_benchmark(file_obj, num_runs: int, batch_sizes_str: str, context_size: int, progress=gr.Progress()):
    """Run benchmark on uploaded GGUF file."""
    if file_obj is None:
        return "❌ Please upload a GGUF file", "", "", None

    # Parse batch sizes
    try:
        batch_sizes = [int(b.strip()) for b in batch_sizes_str.split(',')]
    except:
        return "❌ Invalid batch sizes format. Use comma-separated numbers.", "", "", None

    # Get file path
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

    # Parse GGUF metadata
    progress(0.05, "Parsing model metadata...")
    try:
        metadata = gguf_parser.parse(file_path)
    except Exception as e:
        return f"❌ Failed to parse GGUF file: {e}", "", "", None

    # Add model to database
    db.add_model(metadata)

    # Run benchmarks
    results_text = f"## 📊 Benchmark Results\n\n"
    results_text += f"**Model:** {metadata.name}\n\n"
    results_text += f"**Specs:**\n"
    results_text += f"- Parameters: {metadata.params_billions:.1f}B\n"
    results_text += f"- Quantization: {metadata.quantization}\n"
    results_text += f"- Architecture: {metadata.architecture}\n"
    if metadata.is_moe:
        results_text += f"- MoE: {metadata.moe_experts} experts, {metadata.moe_active_experts} active\n"
    results_text += f"\n---\n\n**Performance:**\n\n"

    def progress_callback(pct, msg):
        progress(0.1 + pct * 0.85, msg)

    try:
        results = benchmark_runner.run(
            file_path, batch_sizes, num_runs, context_size,
            progress_callback=progress_callback
        )
    except Exception as e:
        return f"❌ Benchmark failed: {e}", "", "", None

    # Process results
    all_good = True
    for result in results:
        tps = result['tokens_per_second']
        std = result['std_dev']
        bs = result['batch_size']
        throttle = result['throttling_detected']

        status = "⚠️ Throttling" if throttle else "✅"
        if throttle:
            all_good = False

        results_text += f"- **Batch {bs}:** {tps:.1f} ± {std:.1f} t/s {status}\n"

        # Save to database
        benchmark_result = BenchmarkResult(
            model_hash=metadata.file_hash,
            batch_size=bs,
            tokens_per_second=tps,
            std_dev=std,
            num_runs=result['num_runs'],
            context_size=context_size,
            timestamp=datetime.now().isoformat(),
            throttling_detected=throttle
        )
        db.add_benchmark(benchmark_result)

    # Overall quality
    quality = "✅ GOOD" if all_good else "⚠️ FAIR (throttling detected)"
    results_text += f"\n---\n\n**Overall Quality:** {quality}\n\n"
    results_text += "✅ Added to dataset"

    progress(1.0, "Complete!")

    # Update suggestions and coverage
    suggestions_html = get_smart_suggestions_html()
    coverage_plot = get_coverage_heatmap()
    model_status = get_model_status()

    return results_text, model_status, suggestions_html, coverage_plot

def get_smart_suggestions_html() -> str:
    """Generate HTML for smart suggestions section."""
    df = db.get_all_benchmarks()
    count = len(df)

    if count == 0:
        return "📝 Run your first benchmark to get started!"

    coverage = suggestions.get_coverage_score(df)
    suggestion_list = suggestions.get_suggestions(df, max_suggestions=5)

    html = f"### Coverage Score: {coverage:.0f}%\n\n"

    if not suggestion_list:
        html += "✅ Great coverage! Consider testing edge cases."
        return html

    html += "**Suggested benchmarks:**\n\n"
    for s in suggestion_list:
        stars = "⭐" * s.priority
        html += f"- {stars} **{s.params_billions:.0f}B {s.quantization}** - {s.reason}\n"

    return html

def get_coverage_heatmap():
    """Generate coverage heatmap plot."""
    df = db.get_all_benchmarks()
    matrix = suggestions.get_coverage_matrix(df)

    if matrix.sum().sum() == 0:
        # Empty plot with styled message
        fig = go.Figure()
        fig.add_annotation(
            text="📊 No benchmarks yet<br><br>Run your first benchmark to see coverage!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#667eea"),
            align="center"
        )
        fig.update_layout(
            title=dict(text="📈 Coverage Heatmap", font=dict(size=18, color="#333")),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(102, 126, 234, 0.05)"
        )
        return fig

    fig = px.imshow(
        matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        color_continuous_scale=[[0, "#f8f9fa"], [0.5, "#667eea"], [1, "#764ba2"]],
        labels=dict(x="Parameter Size", y="Quantization", color="Benchmarks"),
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        title=dict(text="📈 Benchmark Coverage Heatmap", font=dict(size=18, color="#333")),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, sans-serif")
    )
    fig.update_traces(textfont=dict(size=12, color="white"))
    return fig

# =============================================================================
# TAB 2: PREDICTOR
# =============================================================================

def predict_performance(model_type: str, params: float, param_unit: str,
                       quant: str, custom_quant: str, batch_size: int,
                       total_params_moe: float, active_params_moe: float,
                       num_experts: int, experts_per_token: int):
    """Make performance prediction."""
    if not predictor.is_trained:
        return "❌ Model not trained yet. Run some benchmarks and click 'Retrain Model' in the Model Performance tab.", "", []

    # Handle parameter units
    if param_unit == "M":
        params = params / 1000
    elif param_unit == "K":
        params = params / 1000000

    # Use custom quant if provided
    if custom_quant and custom_quant.strip():
        quant = custom_quant.strip().upper()

    # Handle MoE
    is_moe = model_type == "MoE"
    moe_experts = 0
    moe_active = 0

    if is_moe:
        params = total_params_moe
        moe_experts = num_experts
        moe_active = experts_per_token

    # Make prediction
    try:
        result = predictor.predict(
            params_billions=params,
            quantization=quant,
            batch_size=batch_size,
            is_moe=is_moe,
            moe_experts=moe_experts,
            moe_active=moe_active
        )
    except Exception as e:
        return f"❌ Prediction failed: {e}", "", []

    # Format results
    pred_text = f"## 🔮 Prediction Results\n\n"
    pred_text += f"# {result.tokens_per_second:.1f} t/s\n\n"
    pred_text += f"**Uncertainty:** ±{result.uncertainty:.1f} t/s (90% CI)\n\n"
    pred_text += f"**Confidence:** {format_confidence_bar(result.confidence_score)}\n\n"

    pred_text += "---\n\n"
    pred_text += "### Details\n\n"
    pred_text += f"- **Ensemble Agreement:** {result.ensemble_agreement:.0f}%\n"
    pred_text += f"- **Neural Net:** {result.nn_prediction:.1f} t/s\n"
    pred_text += f"- **XGBoost:** {result.xgb_prediction:.1f} t/s\n"

    quant_status = "✅ Known" if result.quantization_known else "⚠️ Unknown"
    pred_text += f"- **Quantization:** {quant} {quant_status} ({result.quantization_samples} samples)\n"
    pred_text += f"- **Data Density:** {result.data_density}\n"

    if result.nearest_benchmark:
        nb = result.nearest_benchmark
        pred_text += f"\n**Nearest Benchmark:**\n"
        pred_text += f"- {nb['params']:.1f}B {nb['quant']} @ bs={nb['batch_size']} → {nb['tokens_per_second']:.1f} t/s (actual)\n"
        pred_text += f"- Distance: {result.distance_to_nearest:.2f}\n"

    # Warnings
    if result.warnings:
        pred_text += "\n---\n\n### ⚠️ Warnings\n\n"
        for w in result.warnings:
            pred_text += f"- {w}\n"

    # VRAM estimation
    vram_est = estimate_vram_gb(params, quant)
    vram_available = hardware_profile.vram_gb
    fits = vram_est <= vram_available * 0.95

    pred_text += "\n---\n\n### 💾 Memory Analysis\n\n"
    pred_text += f"- **Estimated VRAM:** {vram_est:.1f} GB ({vram_est/vram_available*100:.0f}%)\n"
    pred_text += f"- **Available VRAM:** {vram_available:.1f} GB\n"
    pred_text += f"- **Fits in VRAM:** {'✅ Yes' if fits else '❌ No'}\n"

    # Batch size scaling
    pred_text += "\n---\n\n### 📈 Batch Size Scaling\n\n"
    scaling = predictor.predict_batch_scaling(
        params, quant, [1, 32, 128, 512],
        is_moe, moe_experts, moe_active
    )
    for s in scaling:
        optimal = " ⭐ Optimal" if s['is_optimal'] else ""
        pred_text += f"- bs={s['batch_size']}: {s['tokens_per_second']:.1f} t/s ({s['pct_of_optimal']:.0f}%){optimal}\n"

    # HuggingFace search
    hf_results = hf_search.search(params_billions=params, quantization=quant, limit=5)

    hf_text = "### 🤗 HuggingFace Models\n\n"
    if hf_results:
        for r in hf_results:
            hf_text += f"**[{r['repo_id']}]({r['url']})**\n"
            hf_text += f"- File: `{r['filename']}`\n"
            hf_text += f"- Downloads: {r['downloads']:,} | Likes: {r['likes']}\n"
            hf_text += f"- [Download]({r['download_url']})\n\n"
    else:
        hf_text += "No matching models found."

    return pred_text, hf_text, hf_results

def update_predictor_visibility(model_type: str):
    """Update visibility of MoE vs Standard inputs."""
    is_moe = model_type == "MoE"
    return (
        gr.update(visible=not is_moe),  # Standard params
        gr.update(visible=is_moe),      # MoE params
    )

# =============================================================================
# TAB 3: DATASET EXPLORER
# =============================================================================

def get_dataset_stats() -> str:
    """Generate dataset statistics markdown."""
    count = db.get_benchmark_count()
    unique_models = db.get_unique_model_count()
    moe_count = db.get_moe_count()
    param_range = db.get_param_range()
    date_range = db.get_date_range()
    quant_counts = db.get_quantization_counts()

    stats = "## 📊 Dataset Statistics\n\n"
    stats += f"- **Total Benchmarks:** {count}\n"
    stats += f"- **Unique Models:** {unique_models}\n"
    stats += f"- **MoE Models:** {moe_count}\n"

    if param_range[1] > 0:
        stats += f"- **Parameter Range:** {param_range[0]:.1f}B - {param_range[1]:.1f}B\n"
    else:
        stats += "- **Parameter Range:** N/A\n"

    stats += f"- **Date Range:** {date_range[0]} to {date_range[1]}\n"

    if quant_counts:
        stats += "\n### Quantization Distribution\n\n"
        for quant, cnt in list(quant_counts.items())[:10]:
            stats += f"- **{quant}:** {cnt} benchmarks\n"

    return stats

def refresh_dataset_stats():
    """Refresh all dataset statistics and plots."""
    stats = get_dataset_stats()
    coverage = get_coverage_heatmap()
    return stats, coverage

# =============================================================================
# TAB 4: MODEL PERFORMANCE
# =============================================================================

def get_model_metrics() -> str:
    """Get ML model performance metrics."""
    if not predictor.is_trained:
        count = db.get_benchmark_count()
        if count < 2:
            return f"⏳ Need at least 2 benchmarks to train. Currently have {count}."
        return "⏳ Model not trained yet. Click 'Retrain Model' below to train."

    metrics = db.get_latest_training_metrics()
    if not metrics:
        return "⏳ No training metrics available."

    # R² star rating
    r2 = metrics['r2']
    if r2 >= 0.95:
        stars = "⭐⭐⭐⭐⭐"
    elif r2 >= 0.90:
        stars = "⭐⭐⭐⭐"
    elif r2 >= 0.80:
        stars = "⭐⭐⭐"
    elif r2 >= 0.60:
        stars = "⭐⭐"
    else:
        stars = "⭐"

    text = "## 🤖 Model Performance Metrics\n\n"
    text += f"- **R² Score:** {r2:.3f} {stars}\n"
    text += f"- **MAE:** {metrics['mae']:.2f} t/s\n"
    text += f"- **RMSE:** {metrics['rmse']:.2f} t/s\n"
    text += f"- **MAPE:** {metrics['mape']:.1f}%\n"
    text += f"- **Samples:** {metrics['num_samples']}\n"
    text += f"- **Last Trained:** {metrics['timestamp'][:19]}\n"

    # Per-quantization breakdown
    df = db.get_all_benchmarks()
    if len(df) >= 3:
        per_quant = predictor.get_per_quantization_metrics(df)
        if per_quant:
            text += "\n---\n\n### Per-Quantization Performance\n\n"
            for quant, m in sorted(per_quant.items(), key=lambda x: -x[1]['r2']):
                text += f"- **{quant}:** R²={m['r2']:.2f}, MAE={m['mae']:.1f} t/s {m['status']} ({m['samples']} samples)\n"

    return text

def retrain_model(progress=gr.Progress()):
    """Retrain the ML model."""
    df = db.get_all_benchmarks()

    if len(df) < 2:
        return f"❌ Need at least 2 benchmarks to train. Currently have {len(df)}."

    progress(0.1, "Preparing training data...")

    try:
        progress(0.3, "Training neural network...")
        metrics = predictor.train(df, epochs=100)
        progress(0.9, "Saving model...")

        # Save metrics to database
        db.save_training_metrics(
            r2=metrics['r2'],
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            num_samples=len(df)
        )

        progress(1.0, "Done!")

        return f"✅ Model retrained successfully!\n\nR² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.2f} t/s"
    except Exception as e:
        return f"❌ Training failed: {e}"

def refresh_model_metrics():
    """Refresh model metrics display."""
    return get_model_metrics()

# =============================================================================
# TAB 5: DATA MANAGEMENT
# =============================================================================

def export_csv():
    """Export dataset to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks_export_{timestamp}.csv"

    try:
        rows = db.export_csv(filename)
        return f"✅ Exported {rows} rows to `{filename}`"
    except Exception as e:
        return f"❌ Export failed: {e}"

# =============================================================================
# TAB 6: HARDWARE
# =============================================================================

def get_hardware_info() -> str:
    """Get hardware profile display."""
    profile = db.get_hardware_profile()
    if not profile:
        profile = hardware_profile

    text = "## 🖥️ Hardware Profile\n\n"
    text += f"- **GPU:** {profile.gpu_name}\n"
    text += f"- **VRAM:** {profile.vram_gb:.1f} GB\n"
    text += f"- **Backend:** {profile.backend}\n"
    text += f"- **System RAM:** {profile.system_ram_gb:.1f} GB\n"
    text += f"- **llama.cpp:** {profile.llama_cpp_version}\n"

    return text

# =============================================================================
# GRADIO APP
# =============================================================================

def create_app():
    """Create the Gradio application."""

    with gr.Blocks(title="LLM Benchmark AI") as app:
        # Custom header with inline styles
        gr.HTML("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 1rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px; color: white;">
            <h1 style="margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                🚀 LLM Benchmark AI
            </h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
                Benchmark GGUF models and predict performance using machine learning
            </p>
        </div>
        """)

        with gr.Tabs():
            # =================================================================
            # TAB 1: BENCHMARK RUNNER
            # =================================================================
            with gr.TabItem("🎯 Benchmark Runner"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload GGUF Model",
                            file_types=[".gguf"],
                            type="filepath"
                        )

                        with gr.Accordion("Configuration", open=True):
                            num_runs = gr.Slider(
                                minimum=1, maximum=10, value=3, step=1,
                                label="Runs per batch size"
                            )
                            batch_sizes = gr.Textbox(
                                value="1,32,128,512",
                                label="Batch sizes (comma-separated)"
                            )
                            context_size = gr.Slider(
                                minimum=512, maximum=32768, value=4096, step=512,
                                label="Context size"
                            )

                        run_btn = gr.Button("▶️ Run Benchmark", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        benchmark_results = gr.Markdown("Upload a model and click Run to start benchmarking.")
                        model_status = gr.Markdown(get_model_status())

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 💡 Smart Suggestions")
                        suggestions_display = gr.Markdown(get_smart_suggestions_html())

                    with gr.Column():
                        coverage_plot = gr.Plot(label="Coverage Heatmap", value=get_coverage_heatmap())

                run_btn.click(
                    fn=run_benchmark,
                    inputs=[file_input, num_runs, batch_sizes, context_size],
                    outputs=[benchmark_results, model_status, suggestions_display, coverage_plot]
                )

            # =================================================================
            # TAB 2: PREDICTOR
            # =================================================================
            with gr.TabItem("🔮 Predictor"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_type = gr.Radio(
                            choices=["Standard", "MoE"],
                            value="Standard",
                            label="Model Type"
                        )

                        # Standard model inputs
                        with gr.Group(visible=True) as standard_inputs:
                            with gr.Row():
                                param_count = gr.Number(value=7, label="Parameters", precision=1)
                                param_unit = gr.Dropdown(
                                    choices=["B", "M", "K"],
                                    value="B",
                                    label="Unit"
                                )

                        # MoE model inputs
                        with gr.Group(visible=False) as moe_inputs:
                            total_params_moe = gr.Number(value=47, label="Total Parameters (B)", precision=1)
                            active_params_moe = gr.Number(value=13, label="Active Parameters (B)", precision=1)
                            num_experts = gr.Number(value=8, label="Number of Experts", precision=0)
                            experts_per_token = gr.Number(value=2, label="Experts per Token", precision=0)

                        # Common inputs
                        quant_choices = db.get_unique_quantizations()
                        if not quant_choices:
                            quant_choices = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"]

                        quant_dropdown = gr.Dropdown(
                            choices=quant_choices,
                            value=quant_choices[0] if quant_choices else "Q4_K_M",
                            label="Quantization"
                        )
                        custom_quant = gr.Textbox(
                            label="Custom Quantization (optional)",
                            placeholder="e.g., IQ4_XS"
                        )
                        batch_size_pred = gr.Number(value=128, label="Batch Size", precision=0)

                        predict_btn = gr.Button("🔮 Predict Performance", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        prediction_results = gr.Markdown("Enter model specs and click Predict.")
                        hf_results_display = gr.Markdown("")

                model_type.change(
                    fn=update_predictor_visibility,
                    inputs=[model_type],
                    outputs=[standard_inputs, moe_inputs]
                )

                hf_results_state = gr.State([])

                predict_btn.click(
                    fn=predict_performance,
                    inputs=[
                        model_type, param_count, param_unit,
                        quant_dropdown, custom_quant, batch_size_pred,
                        total_params_moe, active_params_moe, num_experts, experts_per_token
                    ],
                    outputs=[prediction_results, hf_results_display, hf_results_state]
                )

            # =================================================================
            # TAB 3: DATASET EXPLORER
            # =================================================================
            with gr.TabItem("📊 Dataset Explorer"):
                with gr.Row():
                    with gr.Column():
                        dataset_stats = gr.Markdown(get_dataset_stats())
                        refresh_stats_btn = gr.Button("🔄 Refresh Statistics")

                    with gr.Column():
                        explorer_coverage_plot = gr.Plot(
                            label="Coverage Heatmap",
                            value=get_coverage_heatmap()
                        )

                refresh_stats_btn.click(
                    fn=refresh_dataset_stats,
                    inputs=[],
                    outputs=[dataset_stats, explorer_coverage_plot]
                )

            # =================================================================
            # TAB 4: MODEL PERFORMANCE
            # =================================================================
            with gr.TabItem("🤖 Model Performance"):
                model_metrics = gr.Markdown(get_model_metrics())

                with gr.Row():
                    refresh_metrics_btn = gr.Button("🔄 Refresh Metrics")
                    retrain_btn = gr.Button("🔄 Retrain Model", variant="primary")

                training_status = gr.Markdown("")

                refresh_metrics_btn.click(
                    fn=refresh_model_metrics,
                    inputs=[],
                    outputs=[model_metrics]
                )

                retrain_btn.click(
                    fn=retrain_model,
                    inputs=[],
                    outputs=[training_status]
                ).then(
                    fn=refresh_model_metrics,
                    inputs=[],
                    outputs=[model_metrics]
                )

            # =================================================================
            # TAB 5: DATA MANAGEMENT
            # =================================================================
            with gr.TabItem("💾 Data Management"):
                gr.Markdown("## 💾 Dataset Management")

                with gr.Row():
                    export_btn = gr.Button("📥 Export CSV", variant="primary")

                export_status = gr.Markdown("")

                export_btn.click(
                    fn=export_csv,
                    inputs=[],
                    outputs=[export_status]
                )

                gr.Markdown("""
                ---

                ### Future Features

                - 📤 Import CSV datasets
                - 🔀 Merge datasets from multiple GPUs
                - 📋 Dataset table viewer with filtering
                - 🗑️ Delete individual benchmarks
                """)

            # =================================================================
            # TAB 6: HARDWARE
            # =================================================================
            with gr.TabItem("🖥️ Hardware"):
                hardware_info = gr.Markdown(get_hardware_info())

                gr.Markdown("""
                ---

                ### Notes

                - All benchmarks are tagged with this hardware profile
                - The ML model learns performance characteristics specific to your GPU
                - Predictions are only valid for this hardware configuration
                """)

    return app

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
