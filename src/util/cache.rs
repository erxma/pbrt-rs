use std::{
    collections::HashSet,
    hash::Hash,
    ops::Deref,
    sync::{Arc, Mutex},
};

pub struct InternCache<T> {
    hash_set: Mutex<HashSet<Arc<T>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intern<T>(Arc<T>);

impl<T: Hash + Eq> InternCache<T> {
    pub fn lookup(&mut self, item: T) -> Intern<T> {
        // Creator fn is to just return as is
        self.lookup_with_creator(item, |t| t)
    }

    pub fn lookup_with_creator<F>(&mut self, item: T, create_fn: F) -> Intern<T>
    where
        F: Fn(T) -> T,
    {
        let mut set_guard = self.hash_set.lock().unwrap();
        if let Some(existing) = set_guard.get(&item) {
            Intern(existing.clone())
        } else {
            let new_intern = Arc::new(create_fn(item));
            let arc_clone = new_intern.clone();
            set_guard.insert(new_intern);
            Intern(arc_clone)
        }
    }
}

impl<T> Deref for Intern<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
