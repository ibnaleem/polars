//! See thread: https://lists.apache.org/thread/w88tpz76ox8h3rxkjl4so6rg3f1rv7wt
mod mutable;
mod iterator;
mod view;

use std::any::Any;
use polars_error::*;
use crate::array::Array;
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use std::marker::PhantomData;

mod private {
    pub trait Sealed {}

    impl Sealed for str {}
    impl Sealed for [u8] {}
}
use private::Sealed;

pub trait ViewType: Sealed + 'static {
    const IS_UTF8: bool;

    unsafe fn from_bytes_unchecked(slice: &[u8]) -> &Self;
}

impl ViewType for str {
    const IS_UTF8: bool = true;

    #[inline(always)]
    unsafe fn from_bytes_unchecked(slice: &[u8]) -> &Self {
        std::str::from_utf8_unchecked(slice)
    }
}

impl ViewType for [u8] {
    const IS_UTF8: bool = false;

    #[inline(always)]
    unsafe fn from_bytes_unchecked(slice: &[u8]) -> &Self {
        slice
    }
}

pub type BinaryViewArray = BinaryViewArrayGeneric<[u8]>;
pub type Utf8ViewArray = BinaryViewArrayGeneric<str>;

#[derive(Debug)]
pub struct BinaryViewArrayGeneric<T: ViewType + ?Sized> {
    data_type: ArrowDataType,
    views: Buffer<u128>,
    // Maybe Arc<[Buffer<u8>]>?
    buffers: Vec<Buffer<u8>>,
    // Raw buffer access. (pointer, len).
    raw_buffers: Vec<(*const u8, usize)>,
    validity: Option<Bitmap>,
    phantom: PhantomData<T>
}

impl<T: ViewType + ?Sized> Clone for BinaryViewArrayGeneric<T> {
    fn clone(&self) -> Self {
        Self {
            data_type: self.data_type.clone(),
            views: self.views.clone(),
            buffers: self.buffers.clone(),
            raw_buffers: self.raw_buffers.clone(),
            validity: self.validity.clone(),
            phantom: Default::default()
        }
    }
}

unsafe impl <T: ViewType + ?Sized> Send for BinaryViewArrayGeneric<T> {}
unsafe impl <T: ViewType + ?Sized> Sync for BinaryViewArrayGeneric<T> {}

fn buffers_into_raw<T>(buffers: &[Buffer<T>]) -> Vec<(*const T, usize)> {
    buffers.iter().map(|buf| {
        (buf.as_ptr(), buf.len())
    }).collect()
}

impl<T: ViewType + ?Sized> BinaryViewArrayGeneric<T> {
    /// # Safety
    /// The caller must ensure
    /// - the data is valid utf8 (if required)
    /// - The offsets match the buffers.
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        views: Buffer<u128>,
        buffers: Vec<Buffer<u8>>,
        validity: Option<Bitmap>
    ) -> Self {
        let raw_buffers = buffers_into_raw(&buffers);
        Self {
            data_type,
            views,
            buffers,
            raw_buffers,
            validity,
            phantom: Default::default()
        }
    }

    pub fn try_new(
        data_type: ArrowDataType,
        views: Buffer<u128>,
        buffers: Vec<Buffer<u8>>,
        validity: Option<Bitmap>
    ) -> PolarsResult<Self> {
        if T::IS_UTF8 {
            // check utf8?
            todo!()
        }
        // traverse views and validate offsets?
        if let Some(validity) = &validity {
            polars_ensure!(validity.len()== views.len(), ComputeError: "validity mask length must match the number of values" )
        }

        let raw_buffers = buffers_into_raw(&buffers);
        Ok(Self {
            data_type,
            views,
            buffers,
            raw_buffers,
            validity,
            phantom: Default::default()
        })
    }

    /// Creates an empty [`BinaryViewArrayGeneric`], i.e. whose `.len` is zero.
    #[inline]
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        unsafe { Self::new_unchecked(data_type, Buffer::new(), vec![], None) }
    }

    /// Returns a new null [`BinaryViewArrayGeneric`] of `length`.
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        let validity = Some(Bitmap::new_zeroed(length));
        unsafe {
            Self::new_unchecked(data_type, Buffer::zeroed(length), vec![], validity)
        }
    }

    /// Returns the element at index `i`
    /// # Panics
    /// iff `i >= self.len()`
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i`
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &T {
        let v = self.views.get_unchecked(i);
        let len = *v as u32;

        let bytes = if len < 12 {
            let ptr = self.views.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr.add(i * 16 + 4), len as usize)
        } else {
            self.raw_buffers.get_unchecked()
        }
        todo!()

    }
}


impl<T: ViewType + ?Sized> Array for BinaryViewArrayGeneric<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
        todo!()
    }

    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.views.slice_unchecked(offset, length);
    }

    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        let mut new = self.clone();
        new.validity = validity;
        Box::new(new)
    }

    fn to_boxed(&self) -> Box<dyn Array> {
        Box::new(self.clone())
    }
}