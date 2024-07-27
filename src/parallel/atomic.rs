use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

pub struct AtomicF64 {
    backing_bits: AtomicU64,
}

impl AtomicF64 {
    pub fn new(value: f64) -> Self {
        Self {
            backing_bits: AtomicU64::new(value.to_bits()),
        }
    }

    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.backing_bits.load(ordering))
    }

    pub fn store(&self, value: f64, ordering: Ordering) {
        self.backing_bits.store(value.to_bits(), ordering)
    }

    pub fn fetch_add(&self, value: f64, ordering: Ordering) -> f64 {
        let load_ordering = match ordering {
            Ordering::Relaxed | Ordering::Release => Ordering::Relaxed,
            Ordering::Acquire | Ordering::AcqRel => Ordering::Acquire,
            Ordering::SeqCst => Ordering::SeqCst,
            _ => unimplemented!(),
        };

        let store_ordering = match ordering {
            Ordering::Relaxed | Ordering::Acquire => Ordering::Relaxed,
            Ordering::Release | Ordering::AcqRel => Ordering::Release,
            Ordering::SeqCst => Ordering::SeqCst,
            _ => unimplemented!(),
        };

        loop {
            let old_bits = self.backing_bits.load(load_ordering);
            let old_val = f64::from_bits(old_bits);
            let new_val = old_val + value;
            let new_bits = new_val.to_bits();
            if self
                .backing_bits
                .compare_exchange_weak(old_bits, new_bits, store_ordering, load_ordering)
                .is_ok()
            {
                return old_val;
            }
        }
    }
}

impl Default for AtomicF64 {
    #[inline]
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl From<f64> for AtomicF64 {
    #[inline]
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl fmt::Debug for AtomicF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.load(Ordering::Relaxed).fmt(f)
    }
}
