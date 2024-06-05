use std::{hash::Hash, ops::Deref};

pub struct ArcInternCache<T> {
    inner: inner::ArcInternCache<T>,
}

impl<T: Eq + Hash + Send + Sync + 'static> ArcInternCache<T> {
    pub fn new() -> Self {
        Self {
            inner: inner::ArcInternCache::new(),
        }
    }

    pub fn lookup(&mut self, item: T) -> ArcIntern<T> {
        ArcIntern {
            inner: self.inner.lookup(item),
        }
    }

    pub fn lookup_with_creator<F: Fn(T) -> T>(&mut self, item: T, create_fn: F) -> ArcIntern<T> {
        ArcIntern {
            inner: self.inner.lookup_with_creator(item, create_fn),
        }
    }
}

impl<T: Eq + Hash + Send + Sync + 'static> Default for ArcInternCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct ArcIntern<T: Eq + Hash + Send + Sync + 'static> {
    inner: inner::ArcIntern<T>,
}

impl<T: Hash + Eq + Send + Sync> Deref for ArcIntern<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

trait ArcInternCacheImpl<T> {
    fn new() -> Self;
    fn lookup(&mut self, item: T) -> impl ArcInternImpl<T>;
    fn lookup_with_creator<F: Fn(T) -> T>(
        &mut self,
        item: T,
        create_fn: F,
    ) -> impl ArcInternImpl<T>;
}

trait ArcInternImpl<T>: Deref<Target = T> {}

#[cfg(feature = "internment")]
mod inner {
    pub(super) use internment::ArcIntern;
    use std::{hash::Hash, marker::PhantomData};

    use super::{ArcInternCacheImpl, ArcInternImpl};

    #[derive(Debug)]
    pub(super) struct ArcInternCache<T> {
        phantom: PhantomData<T>,
    }

    impl<T: Hash + Eq + Send + Sync + 'static> ArcInternCacheImpl<T> for ArcInternCache<T> {
        fn new() -> Self {
            Self {
                phantom: PhantomData,
            }
        }

        fn lookup(&mut self, item: T) -> ArcIntern<T> {
            ArcIntern::new(item)
        }

        fn lookup_with_creator<F: Fn(T) -> T>(&mut self, _item: T, _create_fn: F) -> ArcIntern<T> {
            unimplemented!()
        }
    }

    impl<T: Hash + Eq + Send + Sync> ArcInternImpl<T> for ArcIntern<T> {}
}

#[cfg(not(feature = "internment"))]
mod inner {
    use std::{
        collections::HashSet,
        hash::Hash,
        ops::Deref,
        sync::{Arc, Mutex},
    };

    use super::{ArcInternCacheImpl, ArcInternImpl};

    #[derive(Debug)]
    pub(super) struct ArcInternCache<T> {
        hash_set: Mutex<HashSet<Arc<T>>>,
    }

    impl<T: Hash + Eq> ArcInternCacheImpl<T> for ArcInternCache<T> {
        fn new() -> Self {
            Self {
                hash_set: Mutex::new(HashSet::new()),
            }
        }

        fn lookup(&mut self, item: T) -> ArcIntern<T> {
            // Creator fn is to just return as is
            self.lookup_with_creator(item, |t| t)
        }

        fn lookup_with_creator<F: Fn(T) -> T>(&mut self, item: T, create_fn: F) -> ArcIntern<T> {
            let mut set_guard = self.hash_set.lock().unwrap();
            if let Some(existing) = set_guard.get(&item) {
                ArcIntern(existing.clone())
            } else {
                let new_intern = Arc::new(create_fn(item));
                let arc_clone = new_intern.clone();
                set_guard.insert(new_intern);
                ArcIntern(arc_clone)
            }
        }
    }

    #[derive(Debug, Eq, PartialEq)]
    pub(super) struct ArcIntern<T>(Arc<T>);

    impl<T> ArcInternImpl<T> for ArcIntern<T> {}

    impl<T> Deref for ArcIntern<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
}
