//! Python bindings for LSH index

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::PyList;
use krapivin_lsh::{LSHIndex, FileRef, index_parquet_files};
use std::path::PathBuf;

/// File reference returned from LSH queries
#[pyclass]
#[derive(Clone)]
pub struct PyFileRef {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub row_id: u64,
}

#[pymethods]
impl PyFileRef {
    fn __repr__(&self) -> String {
        format!("FileRef(file='{}', row={})", self.file_path, self.row_id)
    }

    fn __str__(&self) -> String {
        format!("{}:{}", self.file_path, self.row_id)
    }
}

impl From<FileRef> for PyFileRef {
    fn from(fr: FileRef) -> Self {
        PyFileRef {
            file_path: fr.file_path,
            row_id: fr.row_id,
        }
    }
}

impl From<&FileRef> for PyFileRef {
    fn from(fr: &FileRef) -> Self {
        PyFileRef {
            file_path: fr.file_path.clone(),
            row_id: fr.row_id,
        }
    }
}

/// LSH index statistics
#[pyclass]
#[derive(Clone)]
pub struct PyIndexStats {
    #[pyo3(get)]
    pub num_buckets: usize,
    #[pyo3(get)]
    pub num_files: usize,
    #[pyo3(get)]
    pub load_factor: f64,
    #[pyo3(get)]
    pub level_densities: Vec<f64>,
}

#[pymethods]
impl PyIndexStats {
    fn __repr__(&self) -> String {
        format!(
            "IndexStats(buckets={}, files={}, load={:.2})",
            self.num_buckets, self.num_files, self.load_factor
        )
    }
}

/// LSH Index for embedding vectors
///
/// Enables fast approximate nearest neighbor search across multiple Parquet files
/// containing embedding vectors from language models.
///
/// # Example
///
/// ```python
/// from krapivin_hash_rs import PyLSHIndex
///
/// # Create index
/// index = PyLSHIndex(
///     seed=12345,
///     num_bits=16,
///     embedding_dim=384,
///     capacity=10000,
///     delta=0.3
/// )
///
/// # Index Parquet files
/// count = index.add_parquet_files(
///     ["embeddings1.parquet", "embeddings2.parquet"],
///     column="embedding"
/// )
///
/// # Query
/// results = index.query([0.1] * 384)
///
/// # Save/load
/// index.save("index.krapivin")
/// index2 = PyLSHIndex.load("index.krapivin")
/// ```
#[pyclass]
pub struct PyLSHIndex {
    index: LSHIndex,
}

#[pymethods]
impl PyLSHIndex {
    /// Create new LSH index
    ///
    /// # Arguments
    /// * `seed` - Fixed seed for LSH hyperplanes (use same seed to merge indexes)
    /// * `num_bits` - Number of LSH hash bits (typically 16-64)
    /// * `embedding_dim` - Dimensionality of embedding vectors
    /// * `capacity` - Hash table capacity
    /// * `delta` - Empty fraction parameter (0 < δ < 1, default 0.3)
    #[new]
    #[pyo3(signature = (seed, num_bits, embedding_dim, capacity, delta=0.3))]
    fn new(
        seed: u64,
        num_bits: usize,
        embedding_dim: usize,
        capacity: usize,
        delta: f64,
    ) -> PyResult<Self> {
        if delta <= 0.0 || delta >= 1.0 {
            return Err(PyValueError::new_err("delta must be in (0, 1)"));
        }
        Ok(PyLSHIndex {
            index: LSHIndex::new(seed, num_bits, embedding_dim, capacity, delta),
        })
    }

    /// Add a single embedding to the index
    ///
    /// # Arguments
    /// * `embedding` - Embedding vector (must match embedding_dim)
    /// * `file_path` - Path to source file
    /// * `row_id` - Row ID in source file
    fn add_embedding(
        &mut self,
        embedding: Vec<f32>,
        file_path: String,
        row_id: u64,
    ) -> PyResult<()> {
        let file_ref = FileRef { file_path, row_id };
        self.index.add_embedding(&embedding, file_ref);
        Ok(())
    }

    /// Index embeddings from Parquet files
    ///
    /// # Arguments
    /// * `file_paths` - List of Parquet file paths
    /// * `column` - Name of column containing embeddings
    ///
    /// # Returns
    /// Total number of embeddings indexed
    fn add_parquet_files(
        &mut self,
        file_paths: Vec<String>,
        column: String,
    ) -> PyResult<usize> {
        let paths: Vec<PathBuf> = file_paths.iter().map(PathBuf::from).collect();

        index_parquet_files(&mut self.index, &paths, &column)
            .map_err(|e| PyIOError::new_err(format!("Failed to index Parquet files: {}", e)))
    }

    /// Query for similar embeddings
    ///
    /// # Arguments
    /// * `embedding` - Query embedding vector
    ///
    /// # Returns
    /// List of FileRef objects in the same LSH bucket, or None if bucket is empty
    fn query(&self, embedding: Vec<f32>) -> PyResult<Option<Vec<PyFileRef>>> {
        let results = self.index.query(&embedding);
        Ok(results.map(|refs| refs.iter().map(PyFileRef::from).collect()))
    }

    /// Get index statistics
    fn stats(&self) -> PyResult<PyIndexStats> {
        let stats = self.index.stats();
        Ok(PyIndexStats {
            num_buckets: stats.num_buckets,
            num_files: stats.num_files,
            load_factor: stats.load_factor,
            level_densities: stats.level_densities,
        })
    }

    /// Get list of indexed files
    fn indexed_files(&self) -> PyResult<Vec<String>> {
        Ok(self.index.indexed_files().to_vec())
    }

    /// Save index to disk
    ///
    /// # Arguments
    /// * `path` - Path to save .krapivin file
    fn save(&self, path: String) -> PyResult<()> {
        self.index
            .save(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to save index: {}", e)))
    }

    /// Load index from disk
    ///
    /// # Arguments
    /// * `path` - Path to .krapivin file
    ///
    /// # Returns
    /// New PyLSHIndex instance
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let index = LSHIndex::load(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to load index: {}", e)))?;
        Ok(PyLSHIndex { index })
    }

    fn __repr__(&self) -> String {
        let stats = self.index.stats();
        format!(
            "PyLSHIndex(buckets={}, files={}, load={:.2})",
            stats.num_buckets, stats.num_files, stats.load_factor
        )
    }

    fn __str__(&self) -> String {
        let stats = self.index.stats();
        format!(
            "LSH Index: {} buckets, {} files indexed, {:.1}% load",
            stats.num_buckets,
            stats.num_files,
            stats.load_factor * 100.0
        )
    }

    /// Get contents of a specific bucket
    ///
    /// # Arguments
    /// * `bucket_id` - LSH bucket ID
    ///
    /// # Returns
    /// List of FileRef objects in this bucket, or None if bucket is empty
    fn get_bucket_contents(&self, bucket_id: u64) -> PyResult<Option<Vec<PyFileRef>>> {
        Ok(self.index.get_bucket(bucket_id)
            .map(|refs| refs.iter().map(PyFileRef::from).collect()))
    }

    /// Get dense buckets above a threshold
    ///
    /// # Arguments
    /// * `min_count` - Minimum number of items in bucket
    ///
    /// # Returns
    /// List of (bucket_id, count) tuples for buckets with >= min_count items
    fn get_dense_buckets(&self, min_count: usize) -> PyResult<Vec<(u64, usize)>> {
        Ok(self.index.dense_buckets(min_count))
    }

    /// Get bucket size distribution
    ///
    /// # Returns
    /// Dictionary mapping bucket_size -> count_of_buckets_with_that_size
    fn bucket_size_histogram(&self) -> PyResult<std::collections::HashMap<usize, usize>> {
        Ok(self.index.bucket_size_histogram())
    }

    /// Iterate over all buckets
    ///
    /// # Returns
    /// List of (bucket_id, count) tuples for all non-empty buckets
    fn all_buckets(&self) -> PyResult<Vec<(u64, usize)>> {
        Ok(self.index.iter_buckets()
            .map(|(bucket_id, refs)| (bucket_id, refs.len()))
            .collect())
    }

    /// Export all index data for Parquet serialization
    ///
    /// # Returns
    /// List of (bucket_id, file_path, row_id) tuples for all indexed embeddings
    fn export_data(&self) -> PyResult<Vec<(u64, String, u64)>> {
        let mut records = Vec::new();

        for (bucket_id, refs) in self.index.iter_buckets() {
            for file_ref in refs {
                records.push((
                    bucket_id,
                    file_ref.file_path.clone(),
                    file_ref.row_id,
                ));
            }
        }

        Ok(records)
    }
}
