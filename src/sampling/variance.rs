use std::ops::AddAssign;

use num_traits::NumCast;

#[derive(Clone, Copy, Debug)]
pub struct VarianceEstimator<T> {
    mean: T,
    sum: T,
    n: usize,
}

impl<T: num_traits::Float> VarianceEstimator<T> {
    pub fn mean(&self) -> T {
        self.mean
    }

    pub fn variance(&self) -> T {
        if self.n > 1 {
            let n: T = NumCast::from(self.n).unwrap();
            self.sum / (n - T::one())
        } else {
            T::zero()
        }
    }

    pub fn relative_variance(&self) -> T {
        if self.n < 1 || self.mean == T::zero() {
            T::zero()
        } else {
            self.variance() / self.mean
        }
    }

    pub fn merge(mut self, ve: Self) -> Self {
        if ve.n == 0 {
            self
        } else {
            let n: T = NumCast::from(self.n).unwrap();
            let ve_n: T = NumCast::from(ve.n).unwrap();
            self.sum = self.sum + ve.sum + (ve.mean - self.mean).sqrt() * n * ve_n / (n + ve_n);
            self.mean = (n * self.mean + ve_n * ve.mean) / (n + ve_n);
            self.n += ve.n;

            self
        }
    }
}

impl<T: num_traits::Float + num_traits::NumAssignOps> AddAssign<T> for VarianceEstimator<T> {
    fn add_assign(&mut self, rhs: T) {
        self.n += 1;
        let delta = rhs - self.mean;
        self.mean += delta / NumCast::from(self.n).unwrap();
        let delta2 = rhs - self.mean;
        self.sum += delta * delta2;
    }
}
