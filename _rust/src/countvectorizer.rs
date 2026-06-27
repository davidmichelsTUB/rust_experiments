use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};

fn sort_vocab_lexi_inplace(vocabulary: &mut FxHashMap<String, usize>, j_indices: &mut Vec<usize>) {
    let n = vocabulary.len();
    let mut sorted: Vec<(&String, usize)> =
        vocabulary.iter().map(|(term, &old)| (term, old)).collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut remap = vec![0usize; n];
    for (new_index, &(_, old_index)) in sorted.iter().enumerate() {
        remap[old_index] = new_index;
    }
    for idx in vocabulary.values_mut() {
        *idx = remap[*idx];
    }
    for col in j_indices.iter_mut() {
        *col = remap[*col];
    }
}

struct Partial {
    vocab: FxHashMap<String, usize>,
    values: Vec<usize>,
    j_indices: Vec<usize>,
    indptr: Vec<usize>,
}
pub fn compute_count_vectorizer_fit(
    corpus: Vec<String>,
    stopwords: FxHashSet<String>,
    tokenizer: regex::Regex,
    n_chunks: usize,
) -> (FxHashMap<String, usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let chunk_size = (corpus.len() / n_chunks).max(1);
    // for reduction
    let identity = || Partial {
        vocab: FxHashMap::default(),
        values: vec![],
        j_indices: vec![],
        indptr: vec![0],
    };

    let mut result = corpus
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut vocab: FxHashMap<String, usize> = FxHashMap::default();

            let mut values: Vec<usize> = Vec::new();
            let mut j_indices: Vec<usize> = Vec::new();
            let mut indptr: Vec<usize> = Vec::with_capacity(chunk.len() + 1);
            indptr.push(0);

            for text in chunk.iter() {
                let mut feature_counter: FxHashMap<usize, usize> = FxHashMap::default();
                for m in tokenizer.find_iter(text) {
                    // to lowercase transforms to String from str
                    let token = m.as_str().to_lowercase();
                    if stopwords.contains(&token) {
                        continue;
                    };
                    let idx = match vocab.get(&token).copied() {
                        Some(i) => i,
                        None => {
                            let i = vocab.len();
                            vocab.insert(token, i);
                            i
                        }
                    };

                    // we need to own the number that our vocabulary returns
                    *feature_counter.entry(idx).or_insert(0) += 1;
                }
                for (&col, &count) in feature_counter.iter() {
                    j_indices.push(col);
                    values.push(count);
                }
                indptr.push(j_indices.len());
            }
            Partial {
                vocab,
                values,
                j_indices,
                indptr,
            }
        })
        .reduce(identity, |mut a, b| {
            let mut remap = vec![0 as usize; b.vocab.len()];
            for (token, &local) in b.vocab.iter() {
                let next = a.vocab.len();
                let global = *a.vocab.entry(token.clone()).or_insert(next);
                remap[local] = global;
            }
            let offset = a.j_indices.len();
            a.values.extend(b.values);
            a.j_indices.extend(b.j_indices.iter().map(|&c| remap[c]));
            a.indptr.extend(b.indptr[1..].iter().map(|&p| p + offset));
            a
        });

    sort_vocab_lexi_inplace(&mut result.vocab, &mut result.j_indices);

    (result.vocab, result.values, result.j_indices, result.indptr)
}

pub fn compute_count_vectorizer_transform(
    corpus: &[String],
    vocabulary: &FxHashMap<String, usize>,
    stopwords: &FxHashSet<String>,
    tokenizer: &Regex,
    n_chunks: usize,
) -> Array2<i32> {
    let n_rows = corpus.len();
    let n_cols = vocabulary.len();
    let chunk_size = (n_rows / n_chunks).max(1);
    let mut out = Array2::<i32>::zeros((n_rows, n_cols));

    out.axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .zip(corpus.par_chunks(chunk_size))
        .for_each(|(mut out_chunk, corpus_chunk)| {
            for (mut out_row, text) in out_chunk.rows_mut().into_iter().zip(corpus_chunk.iter()) {
                for m in tokenizer.find_iter(text) {
                    let token = m.as_str().to_lowercase();
                    if stopwords.contains(&token) {
                        continue;
                    }
                    if let Some(&col) = vocabulary.get(&token) {
                        out_row[col] += 1;
                    }
                }
            }
        });

    out
}

#[pyfunction]
#[pyo3(signature = (corpus, vocabulary, stopwords, token_pattern = r"(?u)\b\w\w+\b".to_string(), n_chunks = 1))]
pub fn count_vectorize_transform(
    py: Python<'_>,
    corpus: Vec<String>,
    vocabulary: FxHashMap<String, usize>,
    stopwords: FxHashSet<String>,
    token_pattern: String,
    n_chunks: usize,
) -> PyResult<Py<PyArray2<i32>>> {
    if n_chunks == 0 {
        return Err(PyValueError::new_err("n_chunks must be >= 1"));
    }
    let tokenizer = Regex::new(&token_pattern)
        .map_err(|e| PyValueError::new_err(format!("invalid token_pattern: {e}")))?;

    let result = py.detach(|| {
        compute_count_vectorizer_transform(&corpus, &vocabulary, &stopwords, &tokenizer, n_chunks)
    });
    Ok(Py::from(result.into_pyarray(py).to_owned()))
}

#[pyfunction]
#[pyo3(signature = (corpus, stopwords, token_pattern = r"(?u)\b\w\w+\b".to_string(), n_chunks = 1))]
pub fn count_vectorize_fit(
    corpus: Vec<String>,
    stopwords: FxHashSet<String>,
    token_pattern: String,
    n_chunks: usize,
) -> PyResult<FxHashMap<String, usize>> {
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }
    let tokenizer = Regex::new(&token_pattern)
        .map_err(|e| PyValueError::new_err(format!("invalid token_pattern: {e}")))?;

    let (vocabulary, _, _, _) =
        compute_count_vectorizer_fit(corpus, stopwords, tokenizer, n_chunks);

    Ok(vocabulary)
}
