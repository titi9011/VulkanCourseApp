#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>
#include <array>

#include "Mesh.h"
#include "Utilities.h"

class VulkanRenderer {
public:
	VulkanRenderer();

	int init(GLFWwindow* newWindow);

	void updateModel(glm::mat4 newModel);

	void draw();
	void cleanup();

	~VulkanRenderer();
	
private:
	GLFWwindow* window;

	int currentFrame = 0;

	// Scene Objects
	std::vector<Mesh> meshList;

	// Scene Settings
	struct MVP {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
	} mvp;


	// Vulkan Component
	// - Main
	VkInstance instance;
	struct {
		VkPhysicalDevice physicalDevice;
		VkDevice logicalDevice;
	} mainDevice;

	VkQueue graphicsQueue;
	VkQueue presentationQueue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapchain;
	std::vector<SwapChainImage> swapChainImages;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	std::vector<VkCommandBuffer> commandBuffers;

	// - Descriptors
	VkDescriptorSetLayout descriptorSetLayout;

	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	std::vector<VkBuffer> uniformBuffer;
	std::vector<VkDeviceMemory> uniformBufferMemory;

	// - Pipeline
	VkPipeline graphicsPipeline;
	VkPipelineLayout pipelineLayout;
	VkRenderPass renderPass;

	// - Pools
	VkCommandPool graphicsCommandPool;

	// - Utility
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;

	// - Synchonisation
	std::vector<VkSemaphore> imageAvailable;
	std::vector<VkSemaphore> renderFinished;
	std::vector<VkFence> drawFences;

	// - Validation
	const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

	// Vulkan Functions
	// - Create Functions
	void createInstance();
	void createLogicalDevice();
	void createSurface();
	void createSwapchain();
	void createRenderPass();
	void createDescriptorSetLayout();
	void createGraphicsPipeline();
	void createFramebuffers();
	void createCommandPool();
	void createCommandBuffers();
	void createSynchronisation();
	
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();

	void updateUniformBuffer(uint32_t imageIndex);

	// - Record Functions
	void recordCommands();

	// - Get Functions
	void getPhysicalDevice();


	// - Support Functions
	// -- Validation
	bool checkValidationLayerSupport();

	// -- Checker Functions
	bool checkInstanceExtensionSupport(std::vector<const char*>* checkExtentions);
	bool checkDeviceExtensionSpupport(VkPhysicalDevice device);
	bool checkDeviceSuitable(VkPhysicalDevice device);

	// -- Getter Functions
	QueueFamilyIndices getQueueFamilies(VkPhysicalDevice device);
	SwapChainDetails getSwapChainDetails(VkPhysicalDevice device);

	// -- Choose Functions
	VkSurfaceFormatKHR chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& format);
	VkPresentModeKHR chooseBestPresentationMode(const std::vector<VkPresentModeKHR>& presentationModes);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities);

	// -- Create Functions
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
	VkShaderModule createShaderModule(const std::vector<char>& code);

};

