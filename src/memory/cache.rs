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
    type Intern: Deref<Target = T>;

    fn new() -> Self;
    fn lookup(&mut self, item: T) -> Self::Intern;
    fn lookup_with_creator<F: Fn(T) -> T>(&mut self, item: T, create_fn: F) -> Self::Intern;
}

mod inner {
    #[cfg(not(feature = "internment"))]
    pub(super) use super::custom_impl::*;
    #[cfg(feature = "internment")]
    pub(super) use super::internment_impl::*;
}

#[cfg(feature = "internment")]
mod internment_impl {
    pub(super) use internment::ArcIntern;
    use std::{hash::Hash, marker::PhantomData};

    use super::ArcInternCacheImpl;

    #[derive(Debug)]
    pub(super) struct ArcInternCache<T> {
        phantom: PhantomData<T>,
    }

    impl<T: Hash + Eq + Send + Sync + 'static> ArcInternCacheImpl<T> for ArcInternCache<T> {
        type Intern = ArcIntern<T>;

        fn new() -> Self {
            Self {
                phantom: PhantomData,
            }
        }

        fn lookup(&mut self, item: T) -> Self::Intern {
            ArcIntern::new(item)
        }

        fn lookup_with_creator<F: Fn(T) -> T>(&mut self, _item: T, _create_fn: F) -> Self::Intern {
            unimplemented!()
        }
    }
}

mod custom_impl {
    use std::{
        collections::HashSet,
        hash::Hash,
        ops::Deref,
        sync::{Arc, Mutex},
    };

    use super::ArcInternCacheImpl;

    #[derive(Debug)]
    #[allow(unused)]
    pub(super) struct ArcInternCache<T> {
        hash_set: Mutex<HashSet<Arc<T>>>,
    }

    impl<T: Hash + Eq> ArcInternCacheImpl<T> for ArcInternCache<T> {
        type Intern = ArcIntern<T>;

        fn new() -> Self {
            Self {
                hash_set: Mutex::new(HashSet::new()),
            }
        }

        fn lookup(&mut self, item: T) -> Self::Intern {
            // Creator fn is to just return as is
            self.lookup_with_creator(item, |t| t)
        }

        fn lookup_with_creator<F: Fn(T) -> T>(&mut self, item: T, create_fn: F) -> Self::Intern {
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

    impl<T> Deref for ArcIntern<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
}
