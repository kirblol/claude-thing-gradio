"""
LLM Benchmark AI - Engine Module
All backend logic: database, GGUF parsing, benchmark runner, ML predictor,
smart suggestions, HuggingFace search, hardware detection.
"""

import sqlite3
import subprocess
import re
import os
import json
import hashlib
import struct
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from huggingface_hub import HfApi, hf_hub_url
import psutil

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelMetadata:
    """Metadata extracted from a GGUF file."""
    name: str
    file_path: str
    file_hash: str
    params_billions: float
    quantization: str
    architecture: str
    is_moe: bool = False
    moe_experts: int = 0
    moe_active_experts: int = 0
    context_length: int = 4096

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_hash: str
    batch_size: int
    tokens_per_second: float
    std_dev: float
    num_runs: int
    context_size: int
    timestamp: str
    throttling_detected: bool = False

@dataclass
class PredictionResult:
    """Result from the ML predictor."""
    tokens_per_second: float
    uncertainty: float
    confidence_score: int  # 1-10
    nn_prediction: float
    xgb_prediction: float
    ensemble_agreement: float
    nearest_benchmark: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)
    quantization_known: bool = True
    quantization_samples: int = 0
    data_density: str = "Unknown"
    distance_to_nearest: float = 0.0

@dataclass
class HardwareProfile:
    """Hardware information."""
    gpu_name: str
    vram_gb: float
    backend: str
    system_ram_gb: float
    llama_cpp_version: str

@dataclass
class SmartSuggestion:
    """A suggested model to benchmark."""
    params_billions: float
    quantization: str
    priority: int  # 1-3 stars
    reason: str
    uncertainty: Optional[float] = None

# =============================================================================
# DATABASE
# =============================================================================

class Database:
    """SQLite database for storing benchmarks and model metadata."""

    def __init__(self, db_path: str = "benchmarks.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # Models table
            c.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    file_hash TEXT PRIMARY KEY,
                    name TEXT,
                    file_path TEXT,
                    params_billions REAL,
                    quantization TEXT,
                    architecture TEXT,
                    is_moe INTEGER DEFAULT 0,
                    moe_experts INTEGER DEFAULT 0,
                    moe_active_experts INTEGER DEFAULT 0,
                    context_length INTEGER DEFAULT 4096,
                    created_at TEXT
                )
            ''')

            # Benchmarks table
            c.execute('''
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_hash TEXT,
                    batch_size INTEGER,
                    tokens_per_second REAL,
                    std_dev REAL,
                    num_runs INTEGER,
                    context_size INTEGER,
                    throttling_detected INTEGER DEFAULT 0,
                    timestamp TEXT,
                    FOREIGN KEY (model_hash) REFERENCES models(file_hash)
                )
            ''')

            # Hardware profile table (single row)
            c.execute('''
                CREATE TABLE IF NOT EXISTS hardware_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    gpu_name TEXT,
                    vram_gb REAL,
                    backend TEXT,
                    system_ram_gb REAL,
                    llama_cpp_version TEXT,
                    updated_at TEXT
                )
            ''')

            # Training history table
            c.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    r2_score REAL,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    num_samples INTEGER
                )
            ''')

            conn.commit()
            conn.close()

    def add_model(self, metadata: ModelMetadata) -> bool:
        """Add a model to the database. Returns True if new, False if exists."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute('SELECT file_hash FROM models WHERE file_hash = ?', (metadata.file_hash,))
            if c.fetchone():
                conn.close()
                return False

            c.execute('''
                INSERT INTO models (file_hash, name, file_path, params_billions, quantization,
                                   architecture, is_moe, moe_experts, moe_active_experts,
                                   context_length, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (metadata.file_hash, metadata.name, metadata.file_path, metadata.params_billions,
                  metadata.quantization, metadata.architecture, int(metadata.is_moe),
                  metadata.moe_experts, metadata.moe_active_experts, metadata.context_length,
                  datetime.now().isoformat()))

            conn.commit()
            conn.close()
            return True

    def add_benchmark(self, result: BenchmarkResult):
        """Add a benchmark result."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute('''
                INSERT INTO benchmarks (model_hash, batch_size, tokens_per_second, std_dev,
                                       num_runs, context_size, throttling_detected, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (result.model_hash, result.batch_size, result.tokens_per_second,
                  result.std_dev, result.num_runs, result.context_size,
                  int(result.throttling_detected), result.timestamp))

            conn.commit()
            conn.close()

    def get_all_benchmarks(self) -> pd.DataFrame:
        """Get all benchmarks joined with model metadata."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT b.*, m.name, m.params_billions, m.quantization, m.architecture,
                       m.is_moe, m.moe_experts, m.moe_active_experts
                FROM benchmarks b
                JOIN models m ON b.model_hash = m.file_hash
                ORDER BY b.timestamp DESC
            ''', conn)
            conn.close()
            return df

    def get_benchmark_count(self) -> int:
        """Get total number of benchmarks."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM benchmarks')
            count = c.fetchone()[0]
            conn.close()
            return count

    def get_unique_model_count(self) -> int:
        """Get number of unique models benchmarked."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(DISTINCT model_hash) FROM benchmarks')
            count = c.fetchone()[0]
            conn.close()
            return count

    def get_unique_quantizations(self) -> List[str]:
        """Get list of unique quantization types in the dataset."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT DISTINCT quantization FROM models ORDER BY quantization')
            quants = [row[0] for row in c.fetchall()]
            conn.close()
            return quants

    def get_quantization_counts(self) -> Dict[str, int]:
        """Get benchmark counts per quantization type."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                SELECT m.quantization, COUNT(*)
                FROM benchmarks b
                JOIN models m ON b.model_hash = m.file_hash
                GROUP BY m.quantization
                ORDER BY COUNT(*) DESC
            ''')
            counts = {row[0]: row[1] for row in c.fetchall()}
            conn.close()
            return counts

    def get_param_range(self) -> Tuple[float, float]:
        """Get min/max parameter counts in dataset."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT MIN(params_billions), MAX(params_billions) FROM models')
            row = c.fetchone()
            conn.close()
            if row and row[0] is not None:
                return (row[0], row[1])
            return (0.0, 0.0)

    def get_moe_count(self) -> int:
        """Get count of MoE models."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM models WHERE is_moe = 1')
            count = c.fetchone()[0]
            conn.close()
            return count

    def get_date_range(self) -> Tuple[str, str]:
        """Get earliest and latest benchmark dates."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM benchmarks')
            row = c.fetchone()
            conn.close()
            if row and row[0]:
                return (row[0][:10], row[1][:10])
            return ("N/A", "N/A")

    def save_hardware_profile(self, profile: HardwareProfile):
        """Save or update hardware profile."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO hardware_profile
                (id, gpu_name, vram_gb, backend, system_ram_gb, llama_cpp_version, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
            ''', (profile.gpu_name, profile.vram_gb, profile.backend,
                  profile.system_ram_gb, profile.llama_cpp_version, datetime.now().isoformat()))
            conn.commit()
            conn.close()

    def get_hardware_profile(self) -> Optional[HardwareProfile]:
        """Get saved hardware profile."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT gpu_name, vram_gb, backend, system_ram_gb, llama_cpp_version FROM hardware_profile')
            row = c.fetchone()
            conn.close()
            if row:
                return HardwareProfile(*row)
            return None

    def save_training_metrics(self, r2: float, mae: float, rmse: float, mape: float, num_samples: int):
        """Save training metrics."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO training_history (timestamp, r2_score, mae, rmse, mape, num_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), r2, mae, rmse, mape, num_samples))
            conn.commit()
            conn.close()

    def get_latest_training_metrics(self) -> Optional[Dict]:
        """Get most recent training metrics."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                SELECT r2_score, mae, rmse, mape, num_samples, timestamp
                FROM training_history
                ORDER BY id DESC LIMIT 1
            ''')
            row = c.fetchone()
            conn.close()
            if row:
                return {
                    'r2': row[0], 'mae': row[1], 'rmse': row[2],
                    'mape': row[3], 'num_samples': row[4], 'timestamp': row[5]
                }
            return None

    def export_csv(self, filepath: str) -> int:
        """Export all benchmarks to CSV. Returns row count."""
        df = self.get_all_benchmarks()
        df.to_csv(filepath, index=False)
        return len(df)

# =============================================================================
# GGUF PARSER
# =============================================================================

class GGUFParser:
    """Parse GGUF file metadata with filename fallback."""

    # Common quantization patterns in filenames
    QUANT_PATTERNS = [
        r'[_.-](Q[0-9]_[A-Z0-9_]+)',  # Q4_K_M, Q5_K_S, etc.
        r'[_.-](Q[0-9]+)',             # Q4, Q8, etc.
        r'[_.-](F16|F32|BF16)',        # Float types
        r'[_.-](IQ[0-9]_[A-Z0-9]+)',   # IQ quantizations
    ]

    # Parameter patterns in filenames
    PARAM_PATTERNS = [
        r'[_.-]([0-9]+\.?[0-9]*)[_.-]?[Bb]',  # 7B, 7.5B, 70B
        r'[_.-]([0-9]+\.?[0-9]*)[Bb][_.-]',   # 7B-
        r'^[^0-9]*([0-9]+\.?[0-9]*)[Bb]',      # starts with name-7B
    ]

    # Architecture hints
    ARCH_PATTERNS = {
        'llama': ['llama', 'alpaca', 'vicuna'],
        'mistral': ['mistral', 'mixtral'],
        'phi': ['phi'],
        'qwen': ['qwen'],
        'gemma': ['gemma'],
        'falcon': ['falcon'],
        'mpt': ['mpt'],
        'starcoder': ['starcoder', 'starcode'],
        'codellama': ['codellama', 'code-llama'],
        'yi': ['yi-'],
        'deepseek': ['deepseek'],
        'command': ['command-r'],
    }

    # MoE indicators
    MOE_PATTERNS = ['moe', 'mixtral', 'switch', 'expert']

    def __init__(self):
        self._gguf_available = self._check_gguf_lib()

    def _check_gguf_lib(self) -> bool:
        """Check if gguf library is available."""
        try:
            import gguf
            return True
        except ImportError:
            return False

    def parse(self, file_path: str) -> ModelMetadata:
        """Parse GGUF file and extract metadata."""
        file_path = os.path.abspath(file_path)
        file_hash = self._compute_hash(file_path)
        filename = os.path.basename(file_path)

        # Try GGUF metadata first
        if self._gguf_available:
            try:
                metadata = self._parse_gguf_metadata(file_path)
                if metadata:
                    metadata.file_hash = file_hash
                    return metadata
            except Exception as e:
                print(f"GGUF parsing failed, falling back to filename: {e}")

        # Fallback to filename parsing
        return self._parse_filename(filename, file_path, file_hash)

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of first 1MB of file (for speed)."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            sha256.update(f.read(1024 * 1024))  # First 1MB
        return sha256.hexdigest()[:16]

    def _parse_gguf_metadata(self, file_path: str) -> Optional[ModelMetadata]:
        """Parse metadata directly from GGUF file."""
        import gguf

        reader = gguf.GGUFReader(file_path)

        metadata = {}
        for field in reader.fields.values():
            if hasattr(field, 'parts'):
                # Handle different field types
                if len(field.parts) > 0:
                    try:
                        if field.types and field.types[0] == gguf.GGUFValueType.STRING:
                            val = str(bytes(field.parts[-1]), 'utf-8')
                        else:
                            val = field.parts[-1].tolist()
                            if isinstance(val, list) and len(val) == 1:
                                val = val[0]
                        metadata[field.name] = val
                    except:
                        pass

        # Extract relevant fields
        name = metadata.get('general.name', os.path.basename(file_path))
        arch = metadata.get('general.architecture', 'unknown')

        # Parameter count
        params = 0
        if 'general.parameter_count' in metadata:
            params = metadata['general.parameter_count'] / 1e9

        # Quantization from filename (GGUF doesn't store this cleanly)
        quant = self._extract_quantization(os.path.basename(file_path))

        # MoE detection
        is_moe = False
        moe_experts = 0
        moe_active = 0

        if 'llama.expert_count' in metadata:
            is_moe = True
            moe_experts = metadata['llama.expert_count']
            moe_active = metadata.get('llama.expert_used_count', moe_experts)
        elif 'model.expert_count' in metadata:
            is_moe = True
            moe_experts = metadata['model.expert_count']
            moe_active = metadata.get('model.expert_used_count', moe_experts)

        # Context length
        ctx_len = metadata.get('llama.context_length',
                              metadata.get('model.context_length', 4096))

        # If params still 0, try to calculate from tensor shapes
        if params == 0:
            params = self._estimate_params_from_filename(os.path.basename(file_path))

        return ModelMetadata(
            name=str(name),
            file_path=file_path,
            file_hash="",  # Will be set by caller
            params_billions=float(params) if params else 0.0,
            quantization=quant,
            architecture=str(arch),
            is_moe=is_moe,
            moe_experts=int(moe_experts),
            moe_active_experts=int(moe_active),
            context_length=int(ctx_len) if isinstance(ctx_len, (int, float)) else 4096
        )

    def _parse_filename(self, filename: str, file_path: str, file_hash: str) -> ModelMetadata:
        """Parse metadata from filename."""
        name = filename.replace('.gguf', '')

        quant = self._extract_quantization(filename)
        params = self._estimate_params_from_filename(filename)
        arch = self._detect_architecture(filename)
        is_moe, moe_experts = self._detect_moe(filename)

        return ModelMetadata(
            name=name,
            file_path=file_path,
            file_hash=file_hash,
            params_billions=params,
            quantization=quant,
            architecture=arch,
            is_moe=is_moe,
            moe_experts=moe_experts,
            moe_active_experts=moe_experts // 4 if is_moe else 0  # Common default
        )

    def _extract_quantization(self, filename: str) -> str:
        """Extract quantization type from filename."""
        filename_upper = filename.upper()
        for pattern in self.QUANT_PATTERNS:
            match = re.search(pattern, filename_upper)
            if match:
                return match.group(1)
        return "UNKNOWN"

    def _estimate_params_from_filename(self, filename: str) -> float:
        """Estimate parameter count from filename."""
        filename_lower = filename.lower()
        for pattern in self.PARAM_PATTERNS:
            match = re.search(pattern, filename_lower, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return 0.0

    def _detect_architecture(self, filename: str) -> str:
        """Detect model architecture from filename."""
        filename_lower = filename.lower()
        for arch, patterns in self.ARCH_PATTERNS.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return arch
        return "unknown"

    def _detect_moe(self, filename: str) -> Tuple[bool, int]:
        """Detect if model is MoE and estimate expert count."""
        filename_lower = filename.lower()

        for pattern in self.MOE_PATTERNS:
            if pattern in filename_lower:
                # Try to extract expert count
                expert_match = re.search(r'(\d+)x\d+', filename_lower)
                if expert_match:
                    return True, int(expert_match.group(1))

                # Mixtral default
                if 'mixtral' in filename_lower:
                    return True, 8

                return True, 8  # Default assumption

        return False, 0

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Run llama-bench and parse results."""

    def __init__(self, llama_bench_path: str = "llama-bench"):
        self.llama_bench_path = llama_bench_path

    def run(self, model_path: str, batch_sizes: List[int], num_runs: int = 3,
            context_size: int = 4096, progress_callback=None) -> List[Dict]:
        """
        Run benchmarks for all batch sizes.
        Returns list of result dicts.
        """
        results = []
        total_tests = len(batch_sizes)

        for i, batch_size in enumerate(batch_sizes):
            if progress_callback:
                progress_callback(i / total_tests, f"Testing batch size {batch_size} ({i+1}/{total_tests})...")

            # Run multiple times and collect results
            run_results = []
            for run in range(num_runs):
                if progress_callback:
                    progress_callback(
                        (i + (run + 1) / num_runs) / total_tests,
                        f"Batch {batch_size}: run {run+1}/{num_runs}..."
                    )

                tps = self._run_single(model_path, batch_size, context_size)
                if tps is not None:
                    run_results.append(tps)

            if run_results:
                mean_tps = np.mean(run_results)
                std_tps = np.std(run_results)

                # Throttling detection: std > 10% of mean
                throttling = std_tps > (mean_tps * 0.1) if mean_tps > 0 else False

                results.append({
                    'batch_size': batch_size,
                    'tokens_per_second': mean_tps,
                    'std_dev': std_tps,
                    'num_runs': len(run_results),
                    'throttling_detected': throttling,
                    'raw_results': run_results
                })

        if progress_callback:
            progress_callback(1.0, "Benchmark complete!")

        return results

    def _run_single(self, model_path: str, batch_size: int, context_size: int) -> Optional[float]:
        """Run a single benchmark and return tokens/second."""
        try:
            cmd = [
                self.llama_bench_path,
                '-m', model_path,
                '-p', '512',  # Prompt tokens
                '-n', '128',  # Generation tokens
                '-b', str(batch_size),
                '-c', str(context_size),
                '-r', '1',    # Single repetition (we handle averaging ourselves)
                '-o', 'json'  # JSON output for easier parsing
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"llama-bench error: {result.stderr}")
                return None

            return self._parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            print("Benchmark timed out")
            return None
        except Exception as e:
            print(f"Benchmark error: {e}")
            return None

    def _parse_output(self, output: str) -> Optional[float]:
        """Parse llama-bench output to extract tokens/second."""
        try:
            # Try JSON parsing first
            lines = output.strip().split('\n')
            for line in lines:
                if line.startswith('[') or line.startswith('{'):
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) > 0:
                        # Look for token generation speed (tg)
                        for entry in data:
                            if 'tg' in entry.get('test', ''):
                                return entry.get('avg_ts', entry.get('t/s'))
                            # Also check 't/s' or 'avg_ts' directly
                            if 'avg_ts' in entry:
                                return entry['avg_ts']
                            if 't/s' in entry:
                                return entry['t/s']

            # Fallback: regex parsing for plain text output
            # Look for patterns like "123.45 tokens/s" or "t/s: 123.45"
            patterns = [
                r'(\d+\.?\d*)\s*tokens?/s',
                r't/s[:\s]+(\d+\.?\d*)',
                r'avg_ts[:\s]+(\d+\.?\d*)',
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Parse error: {e}")
            return None

    def get_version(self) -> str:
        """Get llama-bench version."""
        try:
            result = subprocess.run(
                [self.llama_bench_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() or result.stderr.strip() or "unknown"
        except:
            return "unknown"

# =============================================================================
# ML PREDICTOR
# =============================================================================

class HeteroscedasticMLP(nn.Module):
    """MLP that outputs both mean and variance for uncertainty estimation."""

    def __init__(self, input_dim: int, quant_vocab_size: int = 50, quant_embed_dim: int = 8):
        super().__init__()

        self.quant_embedding = nn.Embedding(quant_vocab_size, quant_embed_dim)

        # Input: [log_params, quant_embed(8), log_batch, is_moe, active_ratio, experts_per_tok]
        # Total: 1 + 8 + 1 + 1 + 1 + 1 = 13
        total_input = 1 + quant_embed_dim + 4

        self.shared = nn.Sequential(
            nn.Linear(total_input, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Log-variance head (log for numerical stability)
        self.logvar_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_continuous: torch.Tensor, quant_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        x_continuous: [batch, 5] - log_params, log_batch, is_moe, active_ratio, experts_per_tok
        quant_idx: [batch] - quantization indices
        """
        quant_embed = self.quant_embedding(quant_idx)  # [batch, 8]
        x = torch.cat([x_continuous[:, :1], quant_embed, x_continuous[:, 1:]], dim=1)

        shared_out = self.shared(x)
        mean = self.mean_head(shared_out)
        logvar = self.logvar_head(shared_out)

        return mean.squeeze(-1), logvar.squeeze(-1)


class MLPredictor:
    """Ensemble predictor using Neural Network + XGBoost."""

    KNOWN_QUANTS = [
        'Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L',
        'Q4_0', 'Q4_1', 'Q4_K_S', 'Q4_K_M',
        'Q5_0', 'Q5_1', 'Q5_K_S', 'Q5_K_M',
        'Q6_K', 'Q8_0',
        'F16', 'F32', 'BF16',
        'IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ2_M',
        'IQ3_XXS', 'IQ3_XS', 'IQ3_S', 'IQ3_M',
        'IQ4_NL', 'IQ4_XS',
        'UNKNOWN'
    ]

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.quant_to_idx = {q: i for i, q in enumerate(self.KNOWN_QUANTS)}
        self.idx_to_quant = {i: q for q, i in self.quant_to_idx.items()}

        self.nn_model: Optional[HeteroscedasticMLP] = None
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()

        self.is_trained = False
        self.training_data: Optional[pd.DataFrame] = None

        self._load_models()

    def _load_models(self):
        """Load saved models if they exist."""
        nn_path = os.path.join(self.model_dir, "nn_model.pt")
        xgb_path = os.path.join(self.model_dir, "xgb_model.json")
        scaler_path = os.path.join(self.model_dir, "scaler.npy")

        if os.path.exists(nn_path) and os.path.exists(xgb_path):
            try:
                self.nn_model = HeteroscedasticMLP(13, len(self.KNOWN_QUANTS))
                self.nn_model.load_state_dict(torch.load(nn_path, weights_only=True))
                self.nn_model.eval()

                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(xgb_path)

                if os.path.exists(scaler_path):
                    scaler_params = np.load(scaler_path, allow_pickle=True).item()
                    self.scaler.mean_ = scaler_params['mean']
                    self.scaler.scale_ = scaler_params['scale']
                    self.scaler.var_ = scaler_params['var']
                    self.scaler.n_features_in_ = scaler_params['n_features']

                self.is_trained = True
            except Exception as e:
                print(f"Error loading models: {e}")
                self.is_trained = False

    def _save_models(self):
        """Save trained models."""
        if self.nn_model:
            torch.save(self.nn_model.state_dict(), os.path.join(self.model_dir, "nn_model.pt"))
        if self.xgb_model:
            self.xgb_model.save_model(os.path.join(self.model_dir, "xgb_model.json"))

        scaler_params = {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_,
            'var': self.scaler.var_,
            'n_features': self.scaler.n_features_in_
        }
        np.save(os.path.join(self.model_dir, "scaler.npy"), scaler_params)

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrices from dataframe."""
        # Continuous features
        log_params = np.log1p(df['params_billions'].values)
        log_batch = np.log1p(df['batch_size'].values)
        is_moe = df['is_moe'].values.astype(float)

        # MoE features - handle potential missing columns
        if 'moe_experts' in df.columns and 'moe_active_experts' in df.columns:
            active_ratio = np.where(
                df['moe_experts'] > 0,
                df['moe_active_experts'] / df['moe_experts'],
                0
            )
            experts_per_tok = df['moe_active_experts'].values / 8  # Normalize
        else:
            active_ratio = np.zeros(len(df))
            experts_per_tok = np.zeros(len(df))

        X_continuous = np.column_stack([log_params, log_batch, is_moe, active_ratio, experts_per_tok])

        # Quantization indices
        quant_idx = np.array([
            self.quant_to_idx.get(q, self.quant_to_idx['UNKNOWN'])
            for q in df['quantization']
        ])

        # Target
        y = df['tokens_per_second'].values

        return X_continuous, quant_idx, y

    def train(self, df: pd.DataFrame, epochs: int = 100) -> Dict[str, float]:
        """Train both models on benchmark data."""
        if len(df) < 10:
            raise ValueError("Need at least 10 benchmarks to train")

        self.training_data = df.copy()

        X_cont, quant_idx, y = self._prepare_features(df)

        # Scale continuous features
        X_cont_scaled = self.scaler.fit_transform(X_cont)

        # Split data
        indices = np.arange(len(df))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        X_train, X_val = X_cont_scaled[train_idx], X_cont_scaled[val_idx]
        q_train, q_val = quant_idx[train_idx], quant_idx[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ===== Train Neural Network =====
        self.nn_model = HeteroscedasticMLP(13, len(self.KNOWN_QUANTS))
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)

        X_train_t = torch.FloatTensor(X_train)
        q_train_t = torch.LongTensor(q_train)
        y_train_t = torch.FloatTensor(y_train)

        dataset = TensorDataset(X_train_t, q_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.nn_model.train()
        for epoch in range(epochs):
            for X_batch, q_batch, y_batch in loader:
                optimizer.zero_grad()

                mean, logvar = self.nn_model(X_batch, q_batch)

                # Negative log-likelihood loss for heteroscedastic regression
                var = torch.exp(logvar) + 1e-6
                loss = 0.5 * (torch.log(var) + (y_batch - mean)**2 / var).mean()

                loss.backward()
                optimizer.step()

        self.nn_model.eval()

        # ===== Train XGBoost =====
        # Combine features for XGBoost (including one-hot quant)
        quant_onehot = np.zeros((len(X_cont_scaled), len(self.KNOWN_QUANTS)))
        quant_onehot[np.arange(len(quant_idx)), quant_idx] = 1
        X_xgb = np.hstack([X_cont_scaled, quant_onehot])

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X_xgb[train_idx], y_train)

        # ===== Evaluate =====
        metrics = self._evaluate(X_val, q_val, y_val, X_xgb[val_idx])

        self.is_trained = True
        self._save_models()

        return metrics

    def _evaluate(self, X_cont: np.ndarray, quant_idx: np.ndarray,
                  y_true: np.ndarray, X_xgb: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        # NN predictions
        with torch.no_grad():
            nn_mean, nn_logvar = self.nn_model(
                torch.FloatTensor(X_cont),
                torch.LongTensor(quant_idx)
            )
            nn_pred = nn_mean.numpy()

        # XGBoost predictions
        xgb_pred = self.xgb_model.predict(X_xgb)

        # Ensemble (simple average for evaluation)
        ensemble_pred = (nn_pred + xgb_pred) / 2

        # Metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        r2 = r2_score(y_true, ensemble_pred)
        mae = mean_absolute_error(y_true, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        mape = np.mean(np.abs((y_true - ensemble_pred) / (y_true + 1e-6))) * 100

        return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}

    def predict(self, params_billions: float, quantization: str, batch_size: int,
                is_moe: bool = False, moe_experts: int = 0,
                moe_active: int = 0) -> PredictionResult:
        """Make a prediction with uncertainty estimation."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Prepare features
        log_params = np.log1p(params_billions)
        log_batch = np.log1p(batch_size)
        active_ratio = moe_active / moe_experts if moe_experts > 0 else 0
        experts_per_tok = moe_active / 8

        X_cont = np.array([[log_params, log_batch, float(is_moe), active_ratio, experts_per_tok]])
        X_cont_scaled = self.scaler.transform(X_cont)

        quant_idx = self.quant_to_idx.get(quantization.upper(), self.quant_to_idx['UNKNOWN'])
        quant_known = quantization.upper() in self.quant_to_idx and quantization.upper() != 'UNKNOWN'

        # NN prediction
        with torch.no_grad():
            nn_mean, nn_logvar = self.nn_model(
                torch.FloatTensor(X_cont_scaled),
                torch.LongTensor([quant_idx])
            )
            nn_pred = nn_mean.item()
            nn_var = np.exp(nn_logvar.item())

        # XGBoost prediction
        quant_onehot = np.zeros((1, len(self.KNOWN_QUANTS)))
        quant_onehot[0, quant_idx] = 1
        X_xgb = np.hstack([X_cont_scaled, quant_onehot])
        xgb_pred = self.xgb_model.predict(X_xgb)[0]

        # Ensemble
        ensemble_pred = (nn_pred + xgb_pred) / 2

        # Agreement measure
        disagreement = abs(nn_pred - xgb_pred)
        agreement = max(0, 100 - (disagreement / ensemble_pred * 100)) if ensemble_pred > 0 else 0

        # Uncertainty: combine NN variance + disagreement
        uncertainty = np.sqrt(nn_var) + disagreement * 0.5

        # Find nearest benchmark
        nearest, distance = self._find_nearest(params_billions, quantization, batch_size, is_moe)

        # Count samples for this quantization
        quant_samples = 0
        if self.training_data is not None:
            quant_samples = len(self.training_data[self.training_data['quantization'] == quantization.upper()])

        # Calculate confidence score (1-10)
        confidence = self._calculate_confidence(
            distance, disagreement, quant_known, quant_samples,
            params_billions, self.training_data
        )

        # Determine data density
        density = self._assess_data_density(params_billions, quantization, batch_size)

        # Generate warnings
        warnings = self._generate_warnings(
            distance, disagreement, quant_known, quant_samples,
            params_billions, batch_size, self.training_data
        )

        return PredictionResult(
            tokens_per_second=ensemble_pred,
            uncertainty=uncertainty,
            confidence_score=confidence,
            nn_prediction=nn_pred,
            xgb_prediction=xgb_pred,
            ensemble_agreement=agreement,
            nearest_benchmark=nearest,
            warnings=warnings,
            quantization_known=quant_known,
            quantization_samples=quant_samples,
            data_density=density,
            distance_to_nearest=distance
        )

    def _find_nearest(self, params: float, quant: str, batch: int,
                      is_moe: bool) -> Tuple[Optional[Dict], float]:
        """Find nearest benchmark in the dataset."""
        if self.training_data is None or len(self.training_data) == 0:
            return None, float('inf')

        df = self.training_data

        # Calculate distances (normalized)
        param_dist = np.abs(np.log1p(df['params_billions']) - np.log1p(params))
        batch_dist = np.abs(np.log1p(df['batch_size']) - np.log1p(batch)) * 0.3
        quant_dist = (df['quantization'] != quant.upper()).astype(float) * 0.5
        moe_dist = (df['is_moe'] != is_moe).astype(float) * 0.3

        total_dist = param_dist + batch_dist + quant_dist + moe_dist
        nearest_idx = total_dist.argmin()

        nearest_row = df.iloc[nearest_idx]
        return {
            'name': nearest_row.get('name', 'Unknown'),
            'params': nearest_row['params_billions'],
            'quant': nearest_row['quantization'],
            'batch_size': nearest_row['batch_size'],
            'tokens_per_second': nearest_row['tokens_per_second']
        }, total_dist.iloc[nearest_idx]

    def _calculate_confidence(self, distance: float, disagreement: float,
                             quant_known: bool, quant_samples: int,
                             params: float, df: Optional[pd.DataFrame]) -> int:
        """Calculate confidence score 1-10."""
        score = 10.0

        # Distance penalty (0-3)
        score -= min(3, distance * 2)

        # Disagreement penalty (0-2)
        if disagreement > 5:
            score -= min(2, disagreement / 5)

        # Unknown quantization penalty (0-2)
        if not quant_known:
            score -= 2
        elif quant_samples < 5:
            score -= 1

        # Extrapolation penalty (0-3)
        if df is not None and len(df) > 0:
            min_params = df['params_billions'].min()
            max_params = df['params_billions'].max()
            if params < min_params * 0.8 or params > max_params * 1.2:
                score -= 3
            elif params < min_params or params > max_params:
                score -= 1.5

        return max(1, min(10, int(round(score))))

    def _assess_data_density(self, params: float, quant: str, batch: int) -> str:
        """Assess data density around the prediction point."""
        if self.training_data is None:
            return "Unknown"

        df = self.training_data

        # Count nearby samples
        param_range = (params * 0.7, params * 1.3)
        nearby = df[
            (df['params_billions'] >= param_range[0]) &
            (df['params_billions'] <= param_range[1])
        ]

        same_quant = nearby[nearby['quantization'] == quant.upper()]

        if len(same_quant) >= 5:
            return f"Good ({len(same_quant)} similar samples)"
        elif len(nearby) >= 10:
            return f"Moderate ({len(nearby)} nearby, {len(same_quant)} same quant)"
        elif len(nearby) >= 3:
            return f"Sparse ({len(nearby)} nearby samples)"
        else:
            return "Very sparse (extrapolating)"

    def _generate_warnings(self, distance: float, disagreement: float,
                          quant_known: bool, quant_samples: int,
                          params: float, batch: int,
                          df: Optional[pd.DataFrame]) -> List[str]:
        """Generate warning messages."""
        warnings = []

        if df is not None and len(df) > 0:
            min_params = df['params_billions'].min()
            max_params = df['params_billions'].max()

            if params < min_params or params > max_params:
                warnings.append(f"⚠️ Extrapolating beyond dataset ({min_params:.1f}B - {max_params:.1f}B)")

        if distance > 1.0:
            warnings.append("⚠️ Far from any benchmarked model")

        if disagreement > 5:
            warnings.append(f"⚠️ Model disagreement: {disagreement:.1f} t/s difference")

        if not quant_known:
            warnings.append("⚠️ Unknown quantization type")
        elif quant_samples < 3:
            warnings.append(f"⚠️ Limited data for {quant_samples} samples of this quantization")

        return warnings

    def get_per_quantization_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get performance metrics broken down by quantization."""
        if not self.is_trained or df is None or len(df) < 10:
            return {}

        results = {}

        for quant in df['quantization'].unique():
            quant_df = df[df['quantization'] == quant]
            if len(quant_df) < 3:
                continue

            X_cont, quant_idx, y_true = self._prepare_features(quant_df)
            X_cont_scaled = self.scaler.transform(X_cont)

            # Predictions
            with torch.no_grad():
                nn_mean, _ = self.nn_model(
                    torch.FloatTensor(X_cont_scaled),
                    torch.LongTensor(quant_idx)
                )
                nn_pred = nn_mean.numpy()

            quant_onehot = np.zeros((len(X_cont_scaled), len(self.KNOWN_QUANTS)))
            quant_onehot[np.arange(len(quant_idx)), quant_idx] = 1
            X_xgb = np.hstack([X_cont_scaled, quant_onehot])
            xgb_pred = self.xgb_model.predict(X_xgb)

            ensemble_pred = (nn_pred + xgb_pred) / 2

            from sklearn.metrics import r2_score, mean_absolute_error

            r2 = r2_score(y_true, ensemble_pred) if len(y_true) > 1 else 0
            mae = mean_absolute_error(y_true, ensemble_pred)

            status = "✅ Excellent" if r2 > 0.9 else "✅ Good" if r2 > 0.8 else "⚠️ Fair" if r2 > 0.6 else "❌ Poor"

            results[quant] = {
                'r2': r2,
                'mae': mae,
                'samples': len(quant_df),
                'status': status
            }

        return results

    def predict_batch_scaling(self, params: float, quant: str,
                              batch_sizes: List[int] = [1, 32, 128, 512],
                              is_moe: bool = False, moe_experts: int = 0,
                              moe_active: int = 0) -> List[Dict]:
        """Predict performance across different batch sizes."""
        results = []

        predictions = []
        for bs in batch_sizes:
            pred = self.predict(params, quant, bs, is_moe, moe_experts, moe_active)
            predictions.append((bs, pred.tokens_per_second))

        # Find optimal
        max_tps = max(p[1] for p in predictions)

        for bs, tps in predictions:
            pct_of_max = (tps / max_tps * 100) if max_tps > 0 else 0
            is_optimal = tps == max_tps
            results.append({
                'batch_size': bs,
                'tokens_per_second': tps,
                'pct_of_optimal': pct_of_max,
                'is_optimal': is_optimal
            })

        return results

# =============================================================================
# SMART SUGGESTIONS
# =============================================================================

class SmartSuggestions:
    """Generate intelligent benchmark suggestions."""

    # Parameter grid (log-spaced)
    PARAM_GRID = [0.5, 1, 1.5, 2, 3, 4, 7, 8, 13, 14, 20, 30, 34, 40, 70, 72, 100, 120]

    # Common quantizations
    QUANT_GRID = ['Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0', 'Q3_K_M', 'Q2_K', 'IQ4_XS']

    def __init__(self, predictor: Optional[MLPredictor] = None):
        self.predictor = predictor

    def get_coverage_score(self, df: pd.DataFrame) -> float:
        """Calculate dataset coverage as a percentage."""
        if df is None or len(df) == 0:
            return 0.0

        covered = set()
        total = len(self.PARAM_GRID) * len(self.QUANT_GRID)

        for _, row in df.iterrows():
            # Find closest param bucket
            params = row['params_billions']
            closest_param = min(self.PARAM_GRID, key=lambda x: abs(x - params))
            quant = row['quantization']

            if quant in self.QUANT_GRID:
                covered.add((closest_param, quant))

        return len(covered) / total * 100

    def get_coverage_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate coverage heatmap data."""
        matrix = pd.DataFrame(
            0,
            index=self.QUANT_GRID,
            columns=[f"{p}B" for p in self.PARAM_GRID]
        )

        if df is None or len(df) == 0:
            return matrix

        for _, row in df.iterrows():
            params = row['params_billions']
            closest_param = min(self.PARAM_GRID, key=lambda x: abs(x - params))
            quant = row['quantization']

            if quant in self.QUANT_GRID:
                col = f"{closest_param}B"
                matrix.loc[quant, col] += 1

        return matrix

    def get_suggestions(self, df: pd.DataFrame, max_suggestions: int = 5) -> List[SmartSuggestion]:
        """Generate prioritized benchmark suggestions."""
        suggestions = []

        if df is None or len(df) == 0:
            # Cold start suggestions
            suggestions.append(SmartSuggestion(
                params_billions=7,
                quantization='Q4_K_M',
                priority=3,
                reason="Popular baseline: 7B models are common and fast to test"
            ))
            suggestions.append(SmartSuggestion(
                params_billions=13,
                quantization='Q4_K_M',
                priority=2,
                reason="Good second point: 13B establishes scaling relationship"
            ))
            return suggestions[:max_suggestions]

        coverage = self.get_coverage_matrix(df)
        existing = set()

        for _, row in df.iterrows():
            params = row['params_billions']
            closest_param = min(self.PARAM_GRID, key=lambda x: abs(x - params))
            existing.add((closest_param, row['quantization']))

        # Priority 1: Interpolation gaps (data on both sides)
        for quant in self.QUANT_GRID:
            params_with_data = sorted([p for p in self.PARAM_GRID if (p, quant) in existing])

            if len(params_with_data) >= 2:
                for i in range(len(params_with_data) - 1):
                    low, high = params_with_data[i], params_with_data[i+1]
                    # Find gaps
                    gaps = [p for p in self.PARAM_GRID if low < p < high and (p, quant) not in existing]
                    for gap in gaps:
                        suggestions.append(SmartSuggestion(
                            params_billions=gap,
                            quantization=quant,
                            priority=3,
                            reason=f"Fills critical gap between {low}B and {high}B"
                        ))

        # Priority 2: High uncertainty regions (if predictor available)
        if self.predictor and self.predictor.is_trained:
            for params in self.PARAM_GRID:
                for quant in self.QUANT_GRID:
                    if (params, quant) in existing:
                        continue
                    try:
                        pred = self.predictor.predict(params, quant, 128)
                        if pred.uncertainty > 5:
                            suggestions.append(SmartSuggestion(
                                params_billions=params,
                                quantization=quant,
                                priority=2,
                                reason=f"High uncertainty region (±{pred.uncertainty:.1f} t/s)",
                                uncertainty=pred.uncertainty
                            ))
                    except:
                        pass

        # Priority 3: Extend coverage (new quant types or param ranges)
        for quant in self.QUANT_GRID:
            quant_count = coverage.loc[quant].sum() if quant in coverage.index else 0
            if quant_count == 0:
                suggestions.append(SmartSuggestion(
                    params_billions=7,  # Start with common size
                    quantization=quant,
                    priority=1,
                    reason=f"New quantization type: no {quant} benchmarks yet"
                ))

        # Sort by priority and dedupe
        suggestions.sort(key=lambda x: (-x.priority, x.uncertainty or 0))

        seen = set()
        unique = []
        for s in suggestions:
            key = (s.params_billions, s.quantization)
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique[:max_suggestions]

# =============================================================================
# HUGGINGFACE SEARCH
# =============================================================================

class HuggingFaceSearch:
    """Search HuggingFace for GGUF models."""

    def __init__(self):
        self.api = HfApi()

    def search(self, params_billions: Optional[float] = None,
               quantization: Optional[str] = None,
               limit: int = 10) -> List[Dict]:
        """Search for GGUF models matching criteria."""

        # Build search query
        query_parts = ['gguf']
        if params_billions:
            # Add common parameter representations
            if params_billions >= 1:
                query_parts.append(f"{params_billions:.0f}B")
            else:
                query_parts.append(f"{params_billions*1000:.0f}M")
        if quantization:
            query_parts.append(quantization)

        query = ' '.join(query_parts)

        try:
            # Search models
            models = self.api.list_models(
                search=query,
                sort="downloads",
                direction=-1,
                limit=limit * 3  # Get more to filter
            )

            results = []
            for model in models:
                # Try to get GGUF files
                try:
                    files = self.api.list_repo_files(model.id)
                    gguf_files = [f for f in files if f.endswith('.gguf')]

                    if not gguf_files:
                        continue

                    # Filter by quantization if specified
                    if quantization:
                        matching = [f for f in gguf_files if quantization.upper() in f.upper()]
                        gguf_files = matching if matching else gguf_files[:1]

                    for gguf_file in gguf_files[:2]:  # Max 2 files per model
                        results.append({
                            'repo_id': model.id,
                            'filename': gguf_file,
                            'downloads': model.downloads or 0,
                            'likes': model.likes or 0,
                            'url': f"https://huggingface.co/{model.id}",
                            'download_url': hf_hub_url(model.id, gguf_file)
                        })

                        if len(results) >= limit:
                            break
                except:
                    continue

                if len(results) >= limit:
                    break

            # Sort by popularity
            results.sort(key=lambda x: x['downloads'], reverse=True)
            return results[:limit]

        except Exception as e:
            print(f"HuggingFace search error: {e}")
            return []

# =============================================================================
# HARDWARE DETECTION
# =============================================================================

class HardwareDetector:
    """Detect system hardware information."""

    def detect(self) -> HardwareProfile:
        """Detect hardware profile."""
        gpu_name, vram_gb, backend = self._detect_gpu()
        system_ram = psutil.virtual_memory().total / (1024**3)
        llama_version = self._get_llama_version()

        return HardwareProfile(
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            backend=backend,
            system_ram_gb=round(system_ram, 1),
            llama_cpp_version=llama_version
        )

    def _detect_gpu(self) -> Tuple[str, float, str]:
        """Detect GPU info."""
        # Try AMD ROCm first
        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Parse ROCm output
                name = "AMD GPU"
                for line in result.stdout.split('\n'):
                    if 'Card series' in line or 'GPU' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            name = parts[1].strip()
                            break

                # Get VRAM
                vram = self._get_rocm_vram()
                return name, vram, "ROCm"
        except:
            pass

        # Try NVIDIA
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                name = parts[0].strip()
                vram_str = parts[1].strip()
                vram = float(re.search(r'(\d+)', vram_str).group(1)) / 1024  # MB to GB
                return name, vram, "CUDA"
        except:
            pass

        return "Unknown GPU", 0.0, "Unknown"

    def _get_rocm_vram(self) -> float:
        """Get VRAM from ROCm."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Total' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            return int(match.group(1)) / (1024**3)  # Bytes to GB
        except:
            pass
        return 0.0

    def _get_llama_version(self) -> str:
        """Get llama.cpp version from llama-bench."""
        try:
            result = subprocess.run(
                ['llama-bench', '--help'],
                capture_output=True, text=True, timeout=5
            )
            # Try to extract version from output
            output = result.stdout + result.stderr
            match = re.search(r'b(\d+)', output)
            if match:
                return f"b{match.group(1)}"
        except:
            pass
        return "unknown"

# =============================================================================
# VRAM ESTIMATION (Simple heuristic, model learns the rest)
# =============================================================================

def estimate_vram_gb(params_billions: float, quantization: str) -> float:
    """Rough VRAM estimate based on params and quantization."""
    # Bits per weight for common quantizations
    bits_per_weight = {
        'F32': 32, 'F16': 16, 'BF16': 16,
        'Q8_0': 8, 'Q6_K': 6.5,
        'Q5_K_M': 5.5, 'Q5_K_S': 5.5, 'Q5_0': 5, 'Q5_1': 5.5,
        'Q4_K_M': 4.5, 'Q4_K_S': 4.5, 'Q4_0': 4, 'Q4_1': 4.5,
        'Q3_K_M': 3.5, 'Q3_K_S': 3.5, 'Q3_K_L': 3.8,
        'Q2_K': 2.5,
        'IQ4_XS': 4.25, 'IQ4_NL': 4.5,
        'IQ3_XXS': 3.0, 'IQ3_XS': 3.25, 'IQ3_S': 3.5, 'IQ3_M': 3.5,
        'IQ2_XXS': 2.0, 'IQ2_XS': 2.25, 'IQ2_S': 2.5, 'IQ2_M': 2.5,
        'IQ1_S': 1.5, 'IQ1_M': 1.75,
    }

    quant_upper = quantization.upper()
    bpw = bits_per_weight.get(quant_upper, 4.5)  # Default to Q4 estimate

    # Basic formula: params * bits / 8 * overhead
    base_gb = params_billions * bpw / 8
    overhead = 1.2  # ~20% overhead for KV cache, etc.

    return base_gb * overhead
