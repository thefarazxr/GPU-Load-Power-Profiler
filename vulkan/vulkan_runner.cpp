#include <vulkan/vulkan.h>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdint>

struct PushConstants { uint32_t iters; uint32_t stride; uint32_t n; uint32_t buf_elems; };

static std::vector<char> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open SPIR-V: " + path);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

static uint32_t find_memory_type(VkPhysicalDevice pdev, uint32_t type_bits,
                                  VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pdev, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("No suitable memory type");
}

static void make_buffer(VkDevice device, VkPhysicalDevice pdev,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags mem_flags,
                        VkBuffer& out_buf, VkDeviceMemory& out_mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &ci, nullptr, &out_buf);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out_buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(pdev, req.memoryTypeBits, mem_flags);
    vkAllocateMemory(device, &ai, nullptr, &out_mem);
    vkBindBufferMemory(device, out_buf, out_mem, 0);
}

void run_vulkan_workload(uint32_t n, uint32_t iters, uint32_t stride,
                         const std::string& spv_path, uint32_t buf_mb) {
    const uint32_t buf_elems = (buf_mb * 1024u * 1024u) / sizeof(uint32_t);
    const VkDeviceSize out_size = n        * sizeof(uint32_t);
    const VkDeviceSize in_size  = buf_elems * sizeof(uint32_t);

    // Instance
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo inst_ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    inst_ci.pApplicationInfo = &app_info;
    VkInstance instance;
    if (vkCreateInstance(&inst_ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");

    // Physical device
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
    std::vector<VkPhysicalDevice> pdevs(dev_count);
    vkEnumeratePhysicalDevices(instance, &dev_count, pdevs.data());
    VkPhysicalDevice pdev = pdevs[0];

    // Queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qf_props.data());
    uint32_t compute_qf = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; ++i)
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { compute_qf = i; break; }
    if (compute_qf == UINT32_MAX) throw std::runtime_error("No compute queue family");

    // Logical device
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q_ci.queueFamilyIndex = compute_qf;
    q_ci.queueCount       = 1;
    q_ci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dev_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos    = &q_ci;
    VkDevice device;
    if (vkCreateDevice(pdev, &dev_ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed");
    VkQueue queue;
    vkGetDeviceQueue(device, compute_qf, 0, &queue);

    // Output buffer (device local, binding 0)
    VkBuffer out_buf; VkDeviceMemory out_mem;
    make_buffer(device, pdev, out_size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                out_buf, out_mem);

    // Input buffer (device local, binding 1) — populated via staging buffer
    VkBuffer in_buf; VkDeviceMemory in_mem;
    make_buffer(device, pdev, in_size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                in_buf, in_mem);

    // Staging buffer — host visible, used once to upload input data
    VkBuffer staging_buf; VkDeviceMemory staging_mem;
    make_buffer(device, pdev, in_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging_buf, staging_mem);

    void* mapped;
    vkMapMemory(device, staging_mem, 0, in_size, 0, &mapped);
    auto* host = static_cast<uint32_t*>(mapped);
    for (uint32_t i = 0; i < buf_elems; ++i) host[i] = i * 2654435761u;
    vkUnmapMemory(device, staging_mem);

    // Command pool
    VkCommandPoolCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp_ci.queueFamilyIndex = compute_qf;
    VkCommandPool cmd_pool;
    vkCreateCommandPool(device, &cp_ci, nullptr, &cmd_pool);

    // Copy staging → device local input buffer
    VkCommandBufferAllocateInfo cb_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cb_ai.commandPool        = cmd_pool;
    cb_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &cb_ai, &cb);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &begin);
    VkBufferCopy copy{0, 0, in_size};
    vkCmdCopyBuffer(cb, staging_buf, in_buf, 1, &copy);
    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fence_ci, nullptr, &fence);
    VkSubmitInfo copy_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers    = &cb;
    vkQueueSubmit(queue, 1, &copy_submit, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &fence);

    // Staging buffer no longer needed
    vkFreeMemory(device, staging_mem, nullptr);
    vkDestroyBuffer(device, staging_buf, nullptr);

    // Shader module
    auto spv = read_spv(spv_path);
    VkShaderModuleCreateInfo sm_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm_ci.codeSize = spv.size();
    sm_ci.pCode    = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule shader;
    vkCreateShaderModule(device, &sm_ci, nullptr, &shader);

    // Descriptor set layout — binding 0: output, binding 1: input
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dsl_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl_ci.bindingCount = 2;
    dsl_ci.pBindings    = bindings;
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &dsl);

    // Push constant range
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset     = 0;
    pc_range.size       = sizeof(PushConstants);

    // Pipeline layout
    VkPipelineLayoutCreateInfo pl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &dsl;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc_range;
    VkPipelineLayout pipeline_layout;
    vkCreatePipelineLayout(device, &pl_ci, nullptr, &pipeline_layout);

    // Compute pipeline
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cp_ci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp_ci2.stage  = stage;
    cp_ci2.layout = pipeline_layout;
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci2, nullptr, &pipeline);

    // Descriptor pool (2 storage buffers) + set
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dp_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp_ci.maxSets       = 1;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes    = &pool_size;
    VkDescriptorPool dp;
    vkCreateDescriptorPool(device, &dp_ci, nullptr, &dp);

    VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds_ai.descriptorPool     = dp;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts        = &dsl;
    VkDescriptorSet ds;
    vkAllocateDescriptorSets(device, &ds_ai, &ds);

    VkDescriptorBufferInfo out_info{out_buf, 0, out_size};
    VkDescriptorBufferInfo in_info {in_buf,  0, in_size};
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = ds;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &out_info;
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = ds;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo     = &in_info;
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    // Compute dispatch
    vkResetCommandPool(device, cmd_pool, 0);
    vkAllocateCommandBuffers(device, &cb_ai, &cb);
    vkBeginCommandBuffer(cb, &begin);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &ds, 0, nullptr);
    PushConstants pc{iters, stride, n, buf_elems};
    vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, (n + 255) / 256, 1, 1);
    vkEndCommandBuffer(cb);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cb;
    vkQueueSubmit(queue, 1, &submit, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyDescriptorPool(device, dp, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkFreeMemory(device, in_mem,  nullptr);
    vkDestroyBuffer(device, in_buf,  nullptr);
    vkFreeMemory(device, out_mem, nullptr);
    vkDestroyBuffer(device, out_buf, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}
