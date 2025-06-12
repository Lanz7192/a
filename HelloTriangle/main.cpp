#include <optional>
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <array>
#include <fstream>
#include <algorithm>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <set>
#define NOMINMAX
#include <Windows.h>

const int MAX_FRAMES_IN_FLIGHT = 2;

using namespace std;

#ifndef NDEBUG
#define DEBUGGING true
#else
#define DEBUGGING false
#endif
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};
#define TRY(v) if(v != VK_SUCCESS){DebugBreak();}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}
const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}
struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value()&&presentFamily.has_value();
	}
	
};
class MyBuffer {
public:
	MyBuffer(VkDevice& device,VkPhysicalDevice& physical, VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags memoryFlags) {
		logicalDevice = device;
		physicalDevice = physical;

		bufferSize = size;

		VkBufferCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		createInfo.size = bufferSize;
		createInfo.usage = usage;
		createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		TRY(vkCreateBuffer(logicalDevice, &createInfo, VK_NULL_HANDLE, &buffer));
		
		vkGetBufferMemoryRequirements(logicalDevice,buffer,&memoryRequirements);

		
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalMemoryProperties);

		uint32_t memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, memoryFlags);

		VkMemoryAllocateInfo memoryAllocation{};
		memoryAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocation.memoryTypeIndex = memoryTypeIndex;
		memoryAllocation.allocationSize = memoryRequirements.size;
		TRY(vkAllocateMemory(logicalDevice, &memoryAllocation, VK_NULL_HANDLE, &memory));
		vkBindBufferMemory(logicalDevice, buffer, memory, 0);
	}
	void map(void** data) {
		TRY(vkMapMemory(logicalDevice, memory, 0, bufferSize, 0,data));
	}
	void unmap(){
		vkUnmapMemory(logicalDevice, memory);
	}
	~MyBuffer() {
		
		destroy();
	}
	void destroy() {
		if (isDestroyed == true)return;
		isDestroyed = true;
		vkDestroyBuffer(logicalDevice, buffer, VK_NULL_HANDLE);
		vkFreeMemory(logicalDevice, memory, VK_NULL_HANDLE);
	}
	const VkBuffer& getBuffer() {
		return buffer;
	}
	MyBuffer(const MyBuffer&) = delete;
	MyBuffer& operator=(const MyBuffer&) = delete;
private:
	uint32_t findMemoryType(uint32_t typeFilter,VkMemoryPropertyFlags memoryFlags) {
		for (uint32_t i = 0;i < physicalMemoryProperties.memoryTypeCount;i++) {
			if ((typeFilter & (1 << i))&&(physicalMemoryProperties.memoryTypes[i].propertyFlags&memoryFlags)==memoryFlags) {
				return i;
			}
		}
		throw runtime_error("Couldn't find memory type!");
	}
	bool isDestroyed = false;
	VkMemoryRequirements memoryRequirements;
	VkPhysicalDeviceMemoryProperties physicalMemoryProperties;
	VkBuffer buffer;
	VkDevice logicalDevice;
	VkDeviceMemory memory;
	VkDeviceSize bufferSize;
	VkPhysicalDevice physicalDevice;
};
class HelloTriangle {
public:
	struct Vertex {
		glm::vec3 position;
		glm::vec3 color;

		static VkVertexInputBindingDescription getBindings() {
			VkVertexInputBindingDescription binding = {};
			binding.binding = 0;
			binding.stride = sizeof(Vertex);
			binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			return binding;
		}

		static array<VkVertexInputAttributeDescription, 2> getAttributes() {
			array<VkVertexInputAttributeDescription, 2> attributes{};

			attributes[0].binding = 0;
			attributes[0].location = 0;
			attributes[0].offset = offsetof(Vertex,position);
			attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributes[1].binding = 0;
			attributes[1].location = 1;
			attributes[1].offset = offsetof(Vertex, color);
			attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			return attributes;
		}
	};
	HelloTriangle() {
		initialiseWindow();


		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createCommandPool();

		createBufferResources();
		addSwapchain();
		createSyncObjects();
	}
	void createBufferResources() {
		vector<Vertex> vertices = {
			{{0,-0.5,0},{1,0,0}},
			{{0.5,0.5,0},{0,1,0}},
			{{-0.5,0.5,0},{0,0,1}}
		};
		vector<uint32_t> indices = {
			0,1,2
		};
		VkDeviceSize vertexBufferSize = 4096;
		VkDeviceSize indexBufferSize = 1024;
		VkDeviceSize totalBufferSize = vertexBufferSize + indexBufferSize;
		stagingBuffer = std::make_unique<MyBuffer>(logicalDevice, physicalDevice, totalBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		vertexBuffer = std::make_unique<MyBuffer>(logicalDevice, physicalDevice, vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		indexBuffer = std::make_unique<MyBuffer>(logicalDevice, physicalDevice, indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		char* bufferData;
		stagingBuffer->map(reinterpret_cast<void**>(&bufferData));
		memcpy(bufferData, vertices.data(), sizeof(Vertex) * vertices.size());
		memcpy(bufferData + vertexBufferSize, indices.data(), sizeof(uint32_t) * indices.size());
		stagingBuffer->unmap();

		copyBuffer(stagingBuffer->getBuffer(), 0, vertexBuffer->getBuffer(), 0, vertexBufferSize);
		copyBuffer(stagingBuffer->getBuffer(), vertexBufferSize, indexBuffer->getBuffer(), 0, indexBufferSize);

	}

	void copyBuffer(VkBuffer srcBuffer,VkDeviceSize srcOffset, VkBuffer dstBuffer,VkDeviceSize dstOffset, VkDeviceSize size) {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		VkBufferCopy copyLocation{};
		copyLocation.dstOffset = dstOffset;
		copyLocation.srcOffset = srcOffset;
		copyLocation.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyLocation);
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		
	}
	
	void createInstance() {
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;
		appInfo.pEngineName = "No Engine";

		VkInstanceCreateInfo instanceInfo{};
		instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceInfo.pApplicationInfo = &appInfo;

		vector<const char*> requiredextensions = getRequiredExtensions();
		instanceInfo.enabledExtensionCount = static_cast<uint32_t>(requiredextensions.size());
		instanceInfo.ppEnabledExtensionNames = requiredextensions.data();
		instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
		instanceInfo.enabledLayerCount = DEBUGGING ? validationLayers.size() : 0;
		instanceInfo.ppEnabledLayerNames = validationLayers.data();
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		populateDebugMessengerCreateInfo(debugCreateInfo);
		instanceInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		TRY(vkCreateInstance(&instanceInfo, VK_NULL_HANDLE, &instance));


		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensionsV(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionsV.data());
		cout << "Available extensions: " << endl;
		for (uint32_t i = 0;i <extensionsV.size();i++) {
			cout << "   " << extensionsV[i].extensionName << endl;
			
		}

		if (!checkValidationLayerSupport()) {
			throw runtime_error("Failed to request validation layers!");
		}
	};	

	~HelloTriangle() {
		destroyAllObjects();
	}

	void run() {
		
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(logicalDevice);
	}


private:
	std::unique_ptr<MyBuffer> vertexBuffer;
	std::unique_ptr<MyBuffer> stagingBuffer;
	std::unique_ptr<MyBuffer> indexBuffer;

	int frameIndex = 0;
	
	VkShaderModule vertexModule;
	VkShaderModule fragmentModule;
	VkPipelineLayout pipelineLayout;
	VkRenderPass renderPass;
	vector<char> readbuffer(const char* name) {
		ifstream in(name, ios::binary);
		in.seekg(0, ios::end);
		uint32_t size = (uint32_t)in.tellg();
		in.seekg(0, ios::beg);

		vector<char> buffer(size);
		in.read(buffer.data(), size);

		return buffer;
	}
	void createRenderPass() {
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}
	void createGraphicsPipeline() {
		vector<char> bufferv = readbuffer("Vertex.spv");
		vector<char> bufferf = readbuffer("Fragment.spv");
		const uint32_t* spirvV = reinterpret_cast<const uint32_t*>(bufferv.data());
		const uint32_t* spirvF = reinterpret_cast<const uint32_t*>(bufferf.data());

		VkShaderModuleCreateInfo moduleFragment{};
		
		moduleFragment.codeSize = bufferf.size();
		moduleFragment.pCode = spirvF;
		moduleFragment.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		VkShaderModuleCreateInfo moduleVertex{};
		
		moduleVertex.codeSize = bufferv.size();
		moduleVertex.pCode = spirvV;
		moduleVertex.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

		TRY(vkCreateShaderModule(logicalDevice, &moduleVertex, VK_NULL_HANDLE,&vertexModule));
		TRY(vkCreateShaderModule(logicalDevice, &moduleFragment, VK_NULL_HANDLE, &fragmentModule));


		
		vector<VkDynamicState> dynStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.pDynamicStates = dynStates.data();
		dynamicState.dynamicStateCount = dynStates.size();

		VkPipelineVertexInputStateCreateInfo inputState{};
		inputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto attributes = Vertex::getAttributes();
		auto binding = Vertex::getBindings();
		inputState.vertexAttributeDescriptionCount = attributes.size();
		inputState.pVertexAttributeDescriptions = attributes.data();
		inputState.vertexBindingDescriptionCount = 1;
		inputState.pVertexBindingDescriptions = &binding;
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		TRY(vkCreatePipelineLayout(logicalDevice, &layoutInfo, VK_NULL_HANDLE, &pipelineLayout));


		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = false;
		rasterizer.rasterizerDiscardEnable = false;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = false;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;

		VkPipelineViewportStateCreateInfo viewportStateInfo{};
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.pScissors = nullptr;
		viewportStateInfo.pViewports = nullptr;

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
		inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyInfo.primitiveRestartEnable = false;
		inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertexModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragmentModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo stages[] = { vertShaderStageInfo, fragShaderStageInfo };
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = stages;
		pipelineInfo.pVertexInputState = &inputState;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportStateInfo;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; // Optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.subpass = 0;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
	}
	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0;i < MAX_FRAMES_IN_FLIGHT;i++) {
			VkSemaphoreCreateInfo semaInfo{};
			VkFenceCreateInfo fenceInfo{};

			semaInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			TRY(vkCreateSemaphore(logicalDevice, &semaInfo, 0, &imageAvailableSemaphores[i]));
			TRY(vkCreateSemaphore(logicalDevice, &semaInfo, 0, &renderFinishedSemaphores[i]));
			TRY(vkCreateFence(logicalDevice, &fenceInfo, 0, &inFlightFences[i]));


		}
	}
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	void addSwapchain() {
		createSwapchain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		
		createCommandBuffer();
	}
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = framebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		VkClearValue clearColor = { {{0.21f, 0.23f, 0.22f, 1.0f}} };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		VkBuffer buffers[] = { vertexBuffer->getBuffer() };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer,0,1,buffers,offsets);
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(commandBuffer, 3, 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}
	void createCommandBuffer() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = commandBuffers.size();

		if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffers[0]) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}
	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}
	void createFramebuffers() {
		framebuffers.resize(swapChainImageViews.size());
		for (int i = 0;i<swapChainImageViews.size();i++) {
			VkImageView attachments[] = {swapChainImageViews[i]};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.format = swapChainImageFormat;
			viewInfo.image = swapChainImages[i];
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			TRY(vkCreateImageView(logicalDevice, &viewInfo, VK_NULL_HANDLE, &swapChainImageViews[i]));
		}
	}
	void createSwapchain() {
		querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(presentModes);
		VkExtent2D extent = chooseSwapExtent(capabilities);

		uint32_t imageCount = capabilities.minImageCount + 1;
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
			imageCount = capabilities.maxImageCount;
		}


		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	};
	void createSurface() {
		TRY(glfwCreateWindowSurface(instance, window, VK_NULL_HANDLE, &surface));
	}
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		float queuePriority = 0.0f;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		vector<VkDeviceQueueCreateInfo> queueInfos;
		for (uint32_t i : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueInfo{};
			queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueInfo.pQueuePriorities = &queuePriority;
			queueInfo.queueFamilyIndex = i;
			queueInfo.queueCount = 1;
			queueInfos.push_back(queueInfo);
		}
		VkPhysicalDeviceFeatures features = {};

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.enabledLayerCount = DEBUGGING ? validationLayers.size() : 0;
		createInfo.ppEnabledLayerNames = DEBUGGING ? validationLayers.data() : nullptr;
		createInfo.pEnabledFeatures = &features;
		createInfo.pQueueCreateInfos = queueInfos.data();
		createInfo.queueCreateInfoCount = queueInfos.size();
		createInfo.enabledExtensionCount = deviceExtensions.size();
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		TRY(vkCreateDevice(physicalDevice, &createInfo, VK_NULL_HANDLE, &logicalDevice));
		vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentQueue);
	}
	void pickPhysicalDevice() {
		uint32_t deviceCount;
		TRY(vkEnumeratePhysicalDevices(instance,&deviceCount,nullptr));
		cout << "device count " << deviceCount << endl;

		vector<VkPhysicalDevice> physicalDevices(deviceCount);
		TRY(vkEnumeratePhysicalDevices(instance, &deviceCount, &physicalDevices[0]));
		physicalDevice = VK_NULL_HANDLE;
		for (auto& device : physicalDevices) {
			if(isDeviceSuitable(device)){ 
				physicalDevice = device;
				break;
			}
		}
		if (physicalDevice == VK_NULL_HANDLE) {
			throw runtime_error("Failed to find suitable physical device");
		}
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice) {
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

		vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, &queueFamilies[0]);

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
			if (presentSupport) {
				indices.presentFamily = i;
			}
			if (indices.isComplete()) {
				
				break;
			}
			i++;
		}

		return indices;
	}

	bool isDeviceSuitable(VkPhysicalDevice physicalDevice) {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		
		bool swapChainAdequate = false;
		
		bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);
		if (extensionsSupported) {
			querySwapChainSupport(physicalDevice);
			return indices.isComplete() && formats.size() != 0 && presentModes.size() != 0;;
		}
		return false;
	}
	void querySwapChainSupport(VkPhysicalDevice physicalDevice) {
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
		if (formatCount > 0) {
			formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, &formats[0]);
		}

		uint32_t presentModeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
		if (presentModeCount > 0) {
			presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, &presentModes[0]);
		}

	}
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}
	void setupDebugMessenger() {
		if (DEBUGGING == false) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);
		createInfo.pUserData = nullptr; // Optional
		
		TRY(CreateDebugUtilsMessengerEXT(instance, &createInfo,VK_NULL_HANDLE,&debugMessenger));
	}
	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		cout << "Available layers:" << endl;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		vector<VkLayerProperties> layers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, &layers[0]);
		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : layers) {
				cout << "    " << layerProperties.layerName << endl;
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		

	}
	vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount;
		const char** extensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		vector<const char*> exts;
		cout << "Required extensions: " << endl;
		for (uint32_t i = 0;i < glfwExtensionCount;i++) {
			
			exts.emplace_back(extensions[i]);
		}
		exts.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		if(DEBUGGING)exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		for (uint32_t i = 0;i < glfwExtensionCount+1+(DEBUGGING?1:0);i++) {
			cout << "   " << exts[i] << endl;
			
		}
		return exts;
	}
	void initialiseWindow() {
		if (!glfwInit()) {
			cout << "Unable to initalize glfw" << endl;
			std::exit(-1);
		}
		
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window = glfwCreateWindow(600, 500, "Vulkan", VK_NULL_HANDLE, VK_NULL_HANDLE);
		if (window == VK_NULL_HANDLE) {
			cout << "Unable to create window" << endl;
			std::exit(-1);
		}
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, resizeCallback);
	}
	bool resized = false;
	static void resizeCallback(GLFWwindow* window, int width, int height) {
		HelloTriangle* app = reinterpret_cast<HelloTriangle*>(glfwGetWindowUserPointer(window));
		app->resized = true;
	}
	void cleanUpSwapchain() {
		for (auto& framebuffer : framebuffers) {
			vkDestroyFramebuffer(logicalDevice, framebuffer, VK_NULL_HANDLE);
		}
		for (auto& imageview : swapChainImageViews) {

			vkDestroyImageView(logicalDevice, imageview, VK_NULL_HANDLE);
		}
		vkDestroySwapchainKHR(logicalDevice, swapChain, VK_NULL_HANDLE);
	}
	void recreateSwapchain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window,&width,&height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkDeviceWaitIdle(logicalDevice);
		cleanUpSwapchain();

		createSwapchain();
		createImageViews();
		createFramebuffers();
	}
	void destroyAllObjects() {
		
		cleanUpSwapchain();
		vkDestroyShaderModule(logicalDevice, vertexModule, 0);
		vkDestroyShaderModule(logicalDevice, fragmentModule, 0);
		for(auto& commandBuffer : commandBuffers)vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
		vkDestroyCommandPool(logicalDevice, commandPool, VK_NULL_HANDLE);
		
		vkDestroyPipeline(logicalDevice, graphicsPipeline, VK_NULL_HANDLE);
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, VK_NULL_HANDLE);
		vkDestroyRenderPass(logicalDevice, renderPass, VK_NULL_HANDLE);
		for (auto& a : imageAvailableSemaphores) {
			vkDestroySemaphore(logicalDevice, a, 0);
		}
		for (auto& a : renderFinishedSemaphores) {
			vkDestroySemaphore(logicalDevice, a, 0);
		}
		for (auto& a : inFlightFences) {
			vkDestroyFence(logicalDevice, a, 0);
		}
		stagingBuffer->destroy();
		vertexBuffer->destroy();
		indexBuffer->destroy();
		vkDestroyDevice(logicalDevice,VK_NULL_HANDLE);

		vkDestroySurfaceKHR(instance, surface, VK_NULL_HANDLE);
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger,VK_NULL_HANDLE);
		vkDestroyInstance(instance,VK_NULL_HANDLE);
	}
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	GLFWwindow* window;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkDevice logicalDevice;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	vector<VkCommandBuffer> commandBuffers;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> framebuffers;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
	vector<VkSemaphore> imageAvailableSemaphores;
	vector<VkSemaphore> renderFinishedSemaphores;
	vector<VkFence> inFlightFences;
	void drawFrame() {
		vkWaitForFences(logicalDevice, 1, &inFlightFences[frameIndex], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = (vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores[frameIndex], VK_NULL_HANDLE, &imageIndex));

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapchain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		vkResetFences(logicalDevice, 1, &inFlightFences[frameIndex]);

		vkResetCommandBuffer(commandBuffers[frameIndex], /*VkCommandBufferResetFlagBits*/ 0);
		recordCommandBuffer(commandBuffers[frameIndex], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[frameIndex] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[frameIndex];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[frameIndex] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[frameIndex]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized) {
			resized = false;
			recreateSwapchain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}
		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}
};
//took 1063 lines
int main() {
	HelloTriangle app;
	app.run();
	return 0;
}
