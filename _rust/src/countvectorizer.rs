use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;

// pub fn csr_to_dense(
//     values: &[usize],
//     indices: &[usize],
//     indptr: &[usize],
//     n_cols: usize,
// ) -> Array2<i32> {
//     let n_rows = indptr.len() - 1;
//     let mut dense = Array2::<i32>::zeros((n_rows, n_cols));

//     let buf = dense
//         .as_slice_mut()
//         .expect("freshly created Array2 is contiguous");

//     for row in 0..n_rows {
//         let base = row * n_cols;
//         for k in indptr[row]..indptr[row + 1] {
//             buf[base + indices[k]] = values[k] as i32;
//         }
//     }

//     dense
// }

fn sort_vocab_lexi_inplace(vocabulary: &mut HashMap<String, usize>, j_indices: &mut Vec<usize>) {
    let n = vocabulary.len();
    let mut sorted: Vec<(&String, usize)> =
        vocabulary.iter().map(|(term, &old)| (term, old)).collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut remap = vec![0usize; n];
    for (new_index, &(_, old_index)) in sorted.iter().enumerate() {
        remap[old_index] = new_index;
    }
    // `sorted` and its borrow of `vocabulary` end here.

    // 3. Rewrite each vocabulary value to its new index. O(V).
    for idx in vocabulary.values_mut() {
        *idx = remap[*idx];
    }

    // 4. Relabel every stored column. O(nnz), plain array lookups.
    for col in j_indices.iter_mut() {
        *col = remap[*col];
    }
}

pub fn compute_count_vectorizer_fit(
    corpus: Vec<String>,
    stopwords: HashSet<String>,
    tokenizer: regex::Regex,
) -> (HashMap<String, usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut vocabulary: HashMap<String, usize, _> = HashMap::new();
    // for reference read https://de.wikipedia.org/wiki/Compressed_Row_Storage
    let mut values: Vec<usize> = Vec::new();
    let mut j_indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = Vec::with_capacity(corpus.len() + 1);
    indptr.push(0);
    for text in corpus.iter() {
        // one feature counter per document
        let mut feature_counter: HashMap<usize, usize> = HashMap::new();
        for m in tokenizer.find_iter(text) {
            // to lowercase transforms to String
            let token = m.as_str().to_lowercase();
            if stopwords.contains(&token) {
                continue;
            };
            let value = vocabulary.get(&token);
            match value {
                Some(_) => {}
                None => {
                    //step for inserting into vocabulary
                    vocabulary.insert(token.clone(), vocabulary.len());
                }
            };
            // we need to own the number that our vocabulary returns
            *feature_counter
                .entry(*vocabulary.get(&token).unwrap())
                .or_insert(0) += 1;
        }
        for (&col, &count) in feature_counter.iter() {
            j_indices.push(col);
            values.push(count);
        }
        indptr.push(j_indices.len());
    }
    sort_vocab_lexi_inplace(&mut vocabulary, &mut j_indices);
    (vocabulary, values, j_indices, indptr)
}

pub fn compute_count_vectorizer_transform(
    corpus: &[String],
    vocabulary: &HashMap<String, usize>,
    stopwords: &HashSet<String>,
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
    vocabulary: HashMap<String, usize>,
    stopwords: HashSet<String>,
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
    stopwords: HashSet<String>,
    token_pattern: String,
    n_chunks: usize,
) -> PyResult<HashMap<String, usize>> {
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_chunks must be >= 1",
        ));
    }
    let tokenizer = Regex::new(&token_pattern)
        .map_err(|e| PyValueError::new_err(format!("invalid token_pattern: {e}")))?;

    let (vocabulary, _, _, _) = compute_count_vectorizer_fit(corpus, stopwords, tokenizer);

    Ok(vocabulary)
}
