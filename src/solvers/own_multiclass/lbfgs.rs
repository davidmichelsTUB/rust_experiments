use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::{compute_loss_multiclass, compute_loss_and_gradient_multiclass, scaled_add2d, dot2d};
use anyhow::{Error};
struct LbfgsState {
    m:      usize,
    s_hist: Vec<Array2<f32>>,
    y_hist: Vec<Array2<f32>>,
    rho:    Vec<f32>,
    head:   usize,
    count:  usize,
}

impl LbfgsState {
    fn new(m: usize, n_classes: usize, n_features: usize) -> Self {
        Self {
            m,
            s_hist: vec![Array2::zeros((n_classes, n_features)); m],
            y_hist: vec![Array2::zeros((n_classes, n_features)); m],
            rho:    vec![0.0f32; m],
            head:   0,
            count:  0,
        }
    }

    fn push(&mut self, s: Array2<f32>, y: Array2<f32>) {
        let ys = dot2d(&y, &s);
        if ys.abs() < 1e-12 { return; }   // skip curvature-violating pairs
        self.rho[self.head]    = ys.recip();
        self.s_hist[self.head] = s;
        self.y_hist[self.head] = y;
        self.head  = (self.head + 1) % self.m;
        self.count += 1;
    }

    /// Two-loop L-BFGS recursion (Nocedal & Wright, Algorithm 7.4).
    /// Returns the descent direction -H⁻¹ g, same shape as grad.
    fn direction(&self, grad: &Array2<f32>) -> Array2<f32> {
        let k = self.count.min(self.m);
        let mut q     = grad.clone();
        let mut alpha = vec![0.0f32; k];

        // first loop: newest → oldest
        for i in (0..k).rev() {
            let idx  = (self.head + self.m - 1 - i) % self.m;
            let a    = self.rho[idx] * dot2d(&self.s_hist[idx], &q);
            alpha[i] = a;
            scaled_add2d(&mut q, -a, &self.y_hist[idx]);
        }

        // initial Hessian scaling γ = sᵀy / yᵀy from most recent pair
        let mut r = if self.count > 0 {
            let idx   = (self.head + self.m - 1) % self.m;
            let sy    = dot2d(&self.s_hist[idx], &self.y_hist[idx]);
            let yy    = dot2d(&self.y_hist[idx], &self.y_hist[idx]);
            let gamma = if yy > 1e-12 { sy / yy } else { 1.0 };
            q.mapv(|v| v * gamma)   // clone + scale in one pass
        } else {
            q
        };

        // second loop: oldest → newest
        for i in 0..k {
            let idx  = (self.head + self.m - 1 - i) % self.m;
            let beta = self.rho[idx] * dot2d(&self.y_hist[idx], &r);
            scaled_add2d(&mut r, alpha[i] - beta, &self.s_hist[idx]);
        }

        r.mapv_inplace(|v| -v);   // negate in-place: descent direction
        r
    }
}

// ── Line search (backtracking Wolfe) ─────────────────────────────────────────

/// Returns `(loss_new, w_new, grad_new)`.
/// Armijo is checked with loss only; gradient is computed only on acceptance.
fn line_search<'a>(
    x:                  ArrayView2<'a, f32>,
    y:                  ArrayView1<'a, usize>,
    w:                  &Array2<f32>,
    direction:          &Array2<f32>,
    f0:                 f32,
    grad0:              &Array2<f32>,
    l1_reg:             f32,
    l2_reg:             f32,
    sample_weights_sum: f32,
    sample_weights:     Option<ArrayView1<'a, f32>>,
) -> (f32, Array2<f32>, Array2<f32>) {
    const C1: f32 = 1e-4;   // Armijo sufficient-decrease constant
    const C2: f32 = 0.9;    // Wolfe curvature constant
    let mut a     = 1.0f32;
    let derphi0   = dot2d(grad0, direction);  // < 0 for descent direction

    for _ in 0..30 {
        let w_new = w + a * direction;

        // ── Armijo check: loss only, no gradient ─────────────────────────────
        let f_new = compute_loss_multiclass(
            x, y, w_new.view(),
            l1_reg, l2_reg, sample_weights_sum, sample_weights,
        );

        if f_new > f0 + C1 * a * derphi0 {
            a *= 0.5;
            continue;   // gradient never computed for rejected steps
        }

        // ── Armijo passed: fused loss+grad for Wolfe curvature check ─────────
        let (f_acc, g_new) = compute_loss_and_gradient_multiclass(
            x, y, w_new.view(),
            l1_reg, l2_reg, sample_weights_sum, sample_weights,
        );

        let derphi = dot2d(&g_new, direction);
        if derphi.abs() <= C2 * derphi0.abs() {
            return (f_acc, w_new, g_new);   // strong Wolfe satisfied
        }
        // curvature condition failed: adjust step
        a = if derphi >= 0.0 { a * 0.5 } else { a * 2.0 };
    }

    // fallback: return best we can do at current a
    let w_new      = w + a * direction;
    let (f_new, g) = compute_loss_and_gradient_multiclass(
        x, y, w_new.view(),
        l1_reg, l2_reg, sample_weights_sum, sample_weights,
    );
    (f_new, w_new, g)
}

////
// ── fit ───────────────────────────────────────────────────────────────────────

pub fn fit_multiclass<'a>(
    x:              ArrayView2<'a, f32>,
    y:              ArrayView1<'a, usize>,
    n_classes:      usize,
    l1_reg:         f32,
    l2_reg:         f32,
    max_iters:      u64,
    m:              usize,
    tolerance:      f32,
    sample_weights: Option<ArrayView1<'a, f32>>,
) -> Result<Array2<f32>, Error> {
    let n_features         = x.ncols();
    let sample_weights_sum = sample_weights
        .map_or(x.nrows() as f32, |sw| sw.sum());

    let mut w     = Array2::<f32>::zeros((n_classes, n_features));
    let mut lbfgs = LbfgsState::new(m, n_classes, n_features);

    // initial loss + gradient
    let (mut loss, mut grad) = compute_loss_and_gradient_multiclass(
        x, y, w.view(),
        l1_reg, l2_reg, sample_weights_sum, sample_weights,
    );

    for _ in 0..max_iters {
        // convergence: gradient L2 norm
        if dot2d(&grad, &grad).sqrt() < tolerance {
            break;
        }

        let direction = lbfgs.direction(&grad);

        let (loss_new, w_new, grad_new) = line_search(
            x, y, &w, &direction, loss, &grad,
            l1_reg, l2_reg, sample_weights_sum, sample_weights,
        );

        // update L-BFGS history before overwriting w and grad
        let s  = &w_new   - &w;
        let yk = &grad_new - &grad;
        lbfgs.push(s, yk);

        w    = w_new;
        grad = grad_new;
        loss = loss_new;   // already computed inside line_search — free
    }

    Ok(w)
}