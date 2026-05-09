// poc_host.cpp — Vulkan host harness for KTQ2_1 dequant POC.
//
// Loads /tmp/poc-ktq2/fixture.bin, uploads the 100×14B blocks to a scalar-
// layout SSBO, dispatches the dequant shader (one workgroup per block,
// requiredSubgroupSize=32), downloads the 100×32 fp32 outputs, and compares
// them against the CUDA-arithmetic reference embedded in the fixture.
//
// Acceptance gate (Layer 3 spec §1.3): max_abs_err ≤ 1e-3, mean_abs_err ≤ 1e-5.
//
// Build:
//   g++ -std=c++17 -O2 poc_host.cpp -o poc_host -lvulkan
//
// Run:
//   glslc -O --target-env=vulkan1.3 \
//         /mnt/d/repos/llama-tq/ggml/src/ggml-vulkan/vulkan-shaders/dequant_ktq2_1.comp \
//         -o /tmp/poc-ktq2/dequant_ktq2_1.spv
//   ./gen_fixture
//   ./poc_host /tmp/poc-ktq2/dequant_ktq2_1.spv /tmp/poc-ktq2/fixture.bin

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "vk error %d at %s:%d\n", (int)r, __FILE__, __LINE__); std::exit(1); } } while (0)

#pragma pack(push, 1)
struct block_ktq2_1 {
    uint16_t d;
    uint8_t  qs[8];
    uint8_t  sb[4];
};
#pragma pack(pop)
static_assert(sizeof(block_ktq2_1) == 14, "host block struct must be 14B");

static std::vector<uint32_t> load_spv(const char * p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", p); std::exit(1); }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint32_t> v(sz / 4);
    f.read((char *)v.data(), sz);
    return v;
}

static uint32_t find_mt(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no matching memory type\n"); std::exit(2);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dequant.spv> <fixture.bin>\n", argv[0]);
        return 1;
    }
    auto spv = load_spv(argv[1]);

    // -------- Load fixture --------
    std::ifstream ff(argv[2], std::ios::binary | std::ios::ate);
    if (!ff) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    size_t fsize = ff.tellg(); ff.seekg(0);
    std::vector<uint8_t> fbuf(fsize); ff.read((char *)fbuf.data(), fsize);
    uint32_t magic = *(uint32_t *)&fbuf[0];
    uint32_t N     = *(uint32_t *)&fbuf[4];
    uint32_t blk_b = *(uint32_t *)&fbuf[8];
    uint32_t epb   = *(uint32_t *)&fbuf[12];
    if (magic != 0x4B544932u || blk_b != 14 || epb != 32) {
        fprintf(stderr, "fixture header mismatch (magic=0x%08x bb=%u epb=%u)\n", magic, blk_b, epb);
        return 1;
    }
    const uint8_t * blocks_p = &fbuf[16];
    const float   * ref_p    = (const float *)(blocks_p + (size_t)N * 14u);
    size_t blocks_bytes = (size_t)N * 14u;
    size_t ref_bytes    = (size_t)N * 32u * sizeof(float);
    fprintf(stderr, "fixture: N=%u  blocks_bytes=%zu  ref_bytes=%zu\n", N, blocks_bytes, ref_bytes);

    // -------- Vulkan setup --------
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t pc;
    vkEnumeratePhysicalDevices(inst, &pc, nullptr);
    std::vector<VkPhysicalDevice> pds(pc);
    vkEnumeratePhysicalDevices(inst, &pc, pds.data());
    VkPhysicalDevice phys = pds[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    fprintf(stderr, "device: %s api=%u.%u.%u\n", props.deviceName,
        VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));

    uint32_t qc;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qc, qp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qc; ++i) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

    float pr = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &pr;

    // Required features:
    //   - subgroupSizeControl + computeFullSubgroups (Vk 1.3 core)
    //   - scalarBlockLayout (Vk 1.2 core)
    //   - storageBuffer8BitAccess + storageBuffer16BitAccess
    //   - shaderInt8 + shaderFloat16
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.scalarBlockLayout = VK_TRUE;
    v12.storageBuffer8BitAccess = VK_TRUE;
    v12.shaderInt8 = VK_TRUE;
    v12.shaderFloat16 = VK_TRUE;
    v12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.subgroupSizeControl = VK_TRUE;
    v13.computeFullSubgroups = VK_TRUE;

    v11.pNext = &v12; v12.pNext = &v13;

    const char * exts[] = {
        "VK_KHR_shader_subgroup_uniform_control_flow",
    };

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &v11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = sizeof(exts) / sizeof(exts[0]);
    dci.ppEnabledExtensionNames = exts;
    VkDevice dev; VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));

    VkQueue q;
    vkGetDeviceQueue(dev, qf, 0, &q);

    // -------- Buffers --------
    auto mk_buf = [&](VkDeviceSize sz) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer b; VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &b));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_mt(phys, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m; VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(dev, b, m, 0));
        void * p; VK_CHECK(vkMapMemory(dev, m, 0, sz, 0, &p));
        return std::tuple<VkBuffer, VkDeviceMemory, void *>(b, m, p);
    };

    auto [in_b, in_m, in_p]   = mk_buf(blocks_bytes);
    auto [out_b, out_m, out_p] = mk_buf(ref_bytes);
    std::memcpy(in_p, blocks_p, blocks_bytes);
    std::memset(out_p, 0xCD, ref_bytes); // sentinel

    // -------- Pipeline --------
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4;
    smci.pCode = spv.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkDescriptorSetLayoutBinding bb[2]{};
    for (int i = 0; i < 2; ++i) {
        bb[i].binding = i;
        bb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bb[i].descriptorCount = 1;
        bb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = bb;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.pNext = &rss;
    cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

    VkDescriptorBufferInfo bi[2]{};
    bi[0].buffer = in_b;  bi[0].range = blocks_bytes;
    bi[1].buffer = out_b; bi[1].range = ref_bytes;
    VkWriteDescriptorSet ww[2]{};
    for (int i = 0; i < 2; ++i) {
        ww[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ww[i].dstSet = ds; ww[i].dstBinding = i;
        ww[i].descriptorCount = 1;
        ww[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ww[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev, 2, ww, 0, nullptr);

    VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpoolci.queueFamilyIndex = qf;
    cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool; VK_CHECK(vkCreateCommandPool(dev, &cpoolci, nullptr, &cpool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cb, N, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(q));

    // -------- Validate --------
    std::vector<float> out(N * 32);
    std::memcpy(out.data(), out_p, ref_bytes);

    double max_err = 0.0, mean_err = 0.0;
    int wrong_lanes = 0, wrong_blocks = 0;
    int worst_block = -1, worst_lane = -1;
    for (uint32_t b = 0; b < N; ++b) {
        bool bad_block = false;
        for (int j = 0; j < 32; ++j) {
            double e = std::fabs((double)out[b * 32 + j] - (double)ref_p[b * 32 + j]);
            mean_err += e;
            if (e > max_err) { max_err = e; worst_block = (int)b; worst_lane = j; }
            if (e > 1e-3) { wrong_lanes++; bad_block = true; }
        }
        if (bad_block) wrong_blocks++;
    }
    mean_err /= (double)(N * 32);

    printf("== KTQ2_1 dequant POC ==\n");
    printf("device: %s\n", props.deviceName);
    printf("blocks: %u  elements/block: 32  total: %u\n", N, N * 32);
    printf("max_abs_err  = %.3e\n", max_err);
    printf("mean_abs_err = %.3e\n", mean_err);
    printf("wrong_lanes (>1e-3)  = %d\n", wrong_lanes);
    printf("wrong_blocks         = %d\n", wrong_blocks);
    if (max_err > 0.0) {
        printf("worst at block=%d lane=%d  shader=%.6f  ref=%.6f\n",
            worst_block, worst_lane,
            out[worst_block * 32 + worst_lane],
            ref_p[worst_block * 32 + worst_lane]);
        // dump full block for debugging
        printf("block %d lanes 0..31:\n", worst_block);
        for (int j = 0; j < 32; ++j) {
            double e = std::fabs((double)out[worst_block*32+j] - (double)ref_p[worst_block*32+j]);
            printf("  [%2d] shader=%+12.6f  ref=%+12.6f  err=%.3e%s\n",
                j, out[worst_block*32+j], ref_p[worst_block*32+j], e, e>1e-3?" <-":"");
        }
    }
    printf("\n");

    bool pass = (max_err <= 1e-3) && (mean_err <= 1e-5);
    printf("ACCEPTANCE: max_abs_err <= 1e-3 (%.3e) AND mean_abs_err <= 1e-5 (%.3e) -> %s\n",
        max_err, mean_err, pass ? "PASS" : "FAIL");

    return pass ? 0 : 2;
}
