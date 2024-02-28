use std::sync::Arc;

use vulkano::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo, RawBuffer},
    command_buffer::ResourceUseRef,
    device::Device,
    image::{sys::RawImage, AllocateImageError, Image, ImageCreateFlags, ImageCreateInfo},
    memory::{
        allocator::{MemoryAllocatorError, MemoryTypeFilter},
        DedicatedAllocation, DeviceMemory, MemoryAllocateInfo, MemoryMapInfo, MemoryRequirements,
        ResourceMemory,
    },
    Validated,
};

pub(crate) trait DeviceExt {
    fn new_image(
        self: Arc<Self>,
        create_info: ImageCreateInfo,
        filter: MemoryTypeFilter,
    ) -> Result<Arc<Image>, Validated<AllocateImageError>>;
    fn new_buffer(
        self: Arc<Self>,
        create_info: BufferCreateInfo,
        filter: MemoryTypeFilter,
    ) -> Result<Arc<Buffer>, Validated<AllocateBufferError>>;
}

fn dedicated_allocation_memory_requirements(
    dedicate_allocation: DedicatedAllocation<'_>,
) -> &'_ MemoryRequirements {
    match dedicate_allocation {
        DedicatedAllocation::Buffer(buffer) => buffer.memory_requirements(),
        DedicatedAllocation::Image(image) => &image.memory_requirements()[0],
    }
}

fn find_memory_type_index(
    device: Arc<Device>,
    memory_type_bits: u32,
    filter: MemoryTypeFilter,
) -> Option<u32> {
    let memory_properties = device.physical_device().memory_properties();
    let MemoryTypeFilter {
        required_flags,
        preferred_flags,
        not_preferred_flags,
    } = filter;
    memory_properties
        .memory_types
        .iter()
        .enumerate()
        .filter(|&(index, memory_type)| {
            (memory_type_bits & (1 << index)) != 0
                && memory_type.property_flags.contains(required_flags)
        })
        .min_by_key(|&(_, memory_type)| {
            (preferred_flags - memory_type.property_flags).count()
                + (memory_type.property_flags & not_preferred_flags).count()
        })
        .map(|(index, _)| index as u32)
}

fn allocate_dedicated(
    device: Arc<Device>,
    dedicate_allocation: DedicatedAllocation<'_>,
    filter: MemoryTypeFilter,
    should_map: bool,
) -> Result<ResourceMemory, MemoryAllocatorError> {
    let memory_requirements = dedicated_allocation_memory_requirements(dedicate_allocation);
    let memory_type_index =
        find_memory_type_index(device.clone(), memory_requirements.memory_type_bits, filter)
            .ok_or(MemoryAllocatorError::FindMemoryType)?;
    let mut device_memory = DeviceMemory::allocate(
        device.clone(),
        MemoryAllocateInfo {
            allocation_size: memory_requirements.layout.size(),
            dedicated_allocation: Some(dedicate_allocation),
            memory_type_index,
            ..Default::default()
        },
    )
    .map_err(MemoryAllocatorError::AllocateDeviceMemory)?;
    if should_map {
        device_memory
            .map(MemoryMapInfo {
                offset: 0,
                size: device_memory.allocation_size(),
                ..Default::default()
            })
            .map_err(MemoryAllocatorError::AllocateDeviceMemory)?;
    }
    Ok(ResourceMemory::new_dedicated(device_memory))
}

impl DeviceExt for Device {
    fn new_image(
        self: Arc<Self>,
        create_info: ImageCreateInfo,
        filter: MemoryTypeFilter,
    ) -> Result<Arc<Image>, Validated<AllocateImageError>> {
        assert!(!create_info.flags.intersects(ImageCreateFlags::DISJOINT));
        let raw_image = RawImage::new(self.clone(), create_info)
            .map_err(|x| x.map(AllocateImageError::CreateImage))?;
        let resource_memory = allocate_dedicated(
            self.clone(),
            DedicatedAllocation::Image(&raw_image),
            filter,
            false,
        )
        .map_err(|x| Validated::Error(AllocateImageError::AllocateMemory(x)))?;
        unsafe { raw_image.bind_memory(Some(resource_memory)) }
            .map_err(|(x, _, _)| x.map(AllocateImageError::BindMemory))
            .map(Arc::new)
    }
    fn new_buffer(
        self: Arc<Self>,
        create_info: BufferCreateInfo,
        filter: MemoryTypeFilter,
    ) -> Result<Arc<Buffer>, Validated<AllocateBufferError>> {
        let buffer = RawBuffer::new(self.clone(), create_info)
            .map_err(|x| x.map(AllocateBufferError::CreateBuffer))?;
        let resource_memory = allocate_dedicated(
            self.clone(),
            DedicatedAllocation::Buffer(&buffer),
            filter,
            true,
        )
        .map_err(|x| Validated::Error(AllocateBufferError::AllocateMemory(x)))?;
        unsafe { buffer.bind_memory(resource_memory) }
            .map_err(|(x, _, _)| x.map(AllocateBufferError::BindMemory))
            .map(Arc::new)
    }
}
