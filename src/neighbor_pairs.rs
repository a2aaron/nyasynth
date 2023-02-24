// Thank you to Cassie for the NeighborPairs implementation!
// NeighborPairs is an iterator allowing for peeking the previously seen element
pub struct NeighborPairs<I>
where
    I: Iterator,
{
    inner: I,
    prev: Option<I::Item>,
}

impl<I> Iterator for NeighborPairs<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().and_then(|item| match self.prev.take() {
            Some(prev) => {
                self.prev = Some(item.clone());
                Some((prev, item))
            }
            None => {
                self.prev = Some(item);
                self.next()
            }
        })
    }
}

pub trait NeighborPairsIter
where
    Self: Iterator + Sized,
{
    fn neighbor_pairs(self) -> NeighborPairs<Self>;
}

impl<I> NeighborPairsIter for I
where
    I: Iterator,
{
    fn neighbor_pairs(self) -> NeighborPairs<Self> {
        NeighborPairs {
            inner: self,
            prev: None,
        }
    }
}
