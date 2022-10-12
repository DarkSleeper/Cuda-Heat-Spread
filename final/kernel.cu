/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "common/book.h"

#include "model_loader.h"
#include "camera.h"

#define MAX_TEMPER 100.0
#define MIN_TEMPER 0.0
#define HEAT_SRC_NUM 10

__constant__ float dev_max_temper[1];
__constant__ float dev_min_temper[1];
__constant__ int dev_heat_src[HEAT_SRC_NUM];

const float pai = 3.1415926f;

const int num_vao = 1;
const int num_vbo = 4;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

GLuint vao[num_vao] = {0};
GLuint vbo[num_vbo] = {0};
GLuint ebo = 0;
cudaGraphicsResource* cuda_vert;
cudaGraphicsResource* cuda_color;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 20.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

// timing
float delta_time = 1.0f;
float last_time = 0.0f;

int vertex_num;
int triangle_vertex_num;

float toRadians(float degrees)
{
	return (degrees * 2.f * pai) / 360.f;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

void init_shader(const char* vertexPath, const char* fragmentPath, GLuint& ID);

void setupVertices(ImportedModel& myModel)
{
	vector<glm::vec3> vert = myModel.getOriginVertices();
	vector<glm::vec2> text = myModel.getTextureCoords();
	vector<glm::vec3> norm = myModel.getNormals();

	vector<float> pValues;
	vector<float> tValues;
	vector<float> nValues;

	vertex_num = myModel.getNumVertices();
	for (int i = 0; i < vertex_num; i++)
	{
		pValues.push_back(vert[i].x);
		pValues.push_back(vert[i].y);
		pValues.push_back(vert[i].z);

		tValues.push_back(text[i].s);
		tValues.push_back(text[i].t);

		nValues.push_back(norm[i].x);
		nValues.push_back(norm[i].y);
		nValues.push_back(norm[i].z);
	}

	auto triangle_indexes = myModel.getTriangleIndexes();
	triangle_vertex_num = triangle_indexes.size();

	glGenVertexArrays(num_vao, vao);
	glBindVertexArray(vao[0]);

	// indexes
	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangle_vertex_num * sizeof(unsigned int), &(triangle_indexes[0]), GL_STATIC_DRAW);

	//vert
	glGenBuffers(num_vbo, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, pValues.size() * sizeof(float), &(pValues[0]), GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&cuda_vert, vbo[0], cudaGraphicsRegisterFlagsNone);

	//uv
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, tValues.size() * sizeof(float), &(tValues[0]), GL_STATIC_DRAW);

	//normal
	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, nValues.size() * sizeof(float), &(nValues[0]), GL_STATIC_DRAW);

	vector<float4> colors;
	colors.resize(vertex_num, float4{0.0f, 1.0f, 0.0f, 1.0f});

	//color
	glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float4), &(colors[0]), GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&cuda_color, vbo[3], cudaGraphicsRegisterFlagsNone);

	// 解绑VAO和VBO， 顺序很重要
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

__global__ void init_temper(int heat_src_num, float* temper) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < heat_src_num) {
		temper[dev_heat_src[index]] = dev_max_temper[0];
	}
}

__global__ void heat_compute(int vertex_num, float3* vertices, int* adj_index, int* adj_array, float* src_temper, float* dst_temper) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < vertex_num) {
		int start = adj_index[index];
		int end = adj_index[index + 1];

		float sum = src_temper[index];
		for (int i = start; i < end; i++) {
			int neighbour = adj_array[i];
			sum += src_temper[neighbour];
		}
		dst_temper[index] = sum / (end - start + 1);
	}
}

__global__ void set_color(int vertex_num, float* temper, float4* color) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < vertex_num) {
		float a = (temper[index] - dev_min_temper[0]) / (dev_max_temper[0] - dev_min_temper[0]);
		if (a > 0.5) {
			color[index].x = 1;
			color[index].y = (1 - a) / a;
		} else {
			color[index].x = a / (1 - a);
			color[index].y = 1;
		}
		color[index].z = 0.f;
		color[index].w = 1.f;
	}
}

void heat_compute(int* dev_adj_index, int* dev_adj_array, float* dev_src_temper, float* dev_dst_temper) {
	// 在CUDA中映射资源，锁定资源
	cudaGraphicsMapResources(1, &cuda_vert, 0);
	cudaGraphicsMapResources(1, &cuda_color, 0);

	size_t size = vertex_num;

	float3* device_vert;
	float4* device_color;
	cudaGraphicsResourceGetMappedPointer((void**)&device_vert, &size, cuda_vert);
	cudaGraphicsResourceGetMappedPointer((void**)&device_color, &size, cuda_color);

	init_temper <<< HEAT_SRC_NUM / 256 + 1, 256 >>> (HEAT_SRC_NUM, dev_src_temper);

	heat_compute<<< vertex_num / 256 + 1, 256 >>>(vertex_num, device_vert, dev_adj_index, dev_adj_array, dev_src_temper, dev_dst_temper);

	set_color<<< vertex_num / 256 + 1, 256 >>> (vertex_num, dev_dst_temper, device_color);

	// 处理完了即可解除资源锁定，OpenGL可以开始利用处理结果了。
	// 注意在CUDA处理过程中，OpenGL如果访问这些锁定的资源会出错。
	cudaGraphicsUnmapResources(1, &cuda_vert, 0);
	cudaGraphicsUnmapResources(1, &cuda_color, 0);
}

int main(void) {

	srand((unsigned)time(NULL));

	GLFWwindow* window;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	// version 4.6 core
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_CORE_PROFILE, GLFW_OPENGL_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "heat", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	//// tell GLFW to capture our mouse
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//set mat
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	auto aspect = (float)width / (float)height;
	glm::mat4 projection_mat = glm::perspective(toRadians(45.f), aspect, 0.01f, 1000.f);
	glm::mat4 model_mat;
	model_mat = glm::identity<glm::mat4>();
	model_mat = glm::translate(model_mat, glm::vec3(0.0f, -200.0f, 0.0f));

	//set shader
	auto vertex_path = "runtime/shader/opacity.vs";
	auto fragment_path = "runtime/shader/opacity.fs";
	GLuint renderingProgram;
	init_shader(vertex_path, fragment_path, renderingProgram);
	
	//set model
	ImportedModel my_model("runtime/model/craneo_low.OBJ");
	setupVertices(my_model);

	//get cuda resources ready
	float max_temper = MAX_TEMPER;
	float min_temper = MIN_TEMPER;
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_max_temper, &max_temper, sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_min_temper, &min_temper, sizeof(float)));

	std::vector<int> heat_src(HEAT_SRC_NUM);
	for (int i = 0; i < HEAT_SRC_NUM; i++) {
		//heat_src[i] = 100 + i;
		heat_src[i] = rand() % vertex_num;
	}
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_heat_src, &heat_src[0], HEAT_SRC_NUM * sizeof(int)));

	//use 1-demention array to present adj: with adj_idx[i] meaning vertice i's neighbours starts at adj_array[adj_idx[i]] place
	auto adj_map = my_model.getAdjMat();
	vector<int> adj_index(vertex_num + 1, 0);
	vector<int>	adj_array;
	for (int i = 0; i < vertex_num; i++) {
		adj_index[i] = adj_array.size();
		for (auto neighbour: adj_map[i]) {
			adj_array.push_back(neighbour);
		}
	}
	//for tail case
	adj_index[vertex_num] = adj_array.size();

	int* dev_adj_index;
	HANDLE_ERROR(cudaMalloc((void**)&dev_adj_index, (vertex_num + 1) * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(dev_adj_index, &adj_index[0], (vertex_num + 1) * sizeof(int), cudaMemcpyHostToDevice));

	int* dev_adj_array;
	HANDLE_ERROR(cudaMalloc((void**)&dev_adj_array, adj_array.size() * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(dev_adj_array, &adj_array[0], adj_array.size() * sizeof(int), cudaMemcpyHostToDevice));

	vector<float> src_temper(vertex_num, MIN_TEMPER);
	float* dev_src_temper;
	HANDLE_ERROR(cudaMalloc((void**)&dev_src_temper, vertex_num * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_src_temper, &src_temper[0], vertex_num * sizeof(float), cudaMemcpyHostToDevice));

	vector<float> dst_temper(vertex_num, MIN_TEMPER);
	float* dev_dst_temper;
	HANDLE_ERROR(cudaMalloc((void**)&dev_dst_temper, vertex_num * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_dst_temper, &dst_temper[0], vertex_num * sizeof(float), cudaMemcpyHostToDevice));


	//set light
	glm::vec3 direct_light = glm::vec3(0, 0, -1);

	//prepare funcs
	auto setMat4 = [&](const std::string& name, const glm::mat4& mat) -> void {
		glUniformMatrix4fv(glGetUniformLocation(renderingProgram, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	};
	auto setVec3 = [&](const std::string& name, const glm::vec3& value) -> void {
		glUniform3fv(glGetUniformLocation(renderingProgram, name.c_str()), 1, &value[0]);
	};

	bool use_src = true;
	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		//if (last_time == 0.0) {
		//	delta_time = 0.0;
		//	last_time = current_time;
		//} else {
		//	delta_time = current_time - last_time;
		//	//std::cout<<"delta_time:"<<delta_time<<std::endl;
		//	last_time = current_time;
		//}
		processInput(window);
		glm::mat4 view_mat = camera.GetViewMatrix();
		glm::mat4 inv_world_mat = glm::inverse(view_mat);

		/* Cuda here */
		//dynamically use gl resource through cuda to change its values
		if (use_src) {
			heat_compute(dev_adj_index, dev_adj_array, dev_src_temper, dev_dst_temper);
		} else {
			heat_compute(dev_adj_index, dev_adj_array, dev_dst_temper, dev_src_temper);
		}
		use_src = !use_src;

		/* Render here */
		auto current_time = (float)glfwGetTime();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.f);

		//启动着色器程序,在GPU上安装GLSL代码,这不会运行着色器程序，
		glUseProgram(renderingProgram);

		glBindVertexArray(vao[0]);

		setMat4("view_to_clip_matrix", projection_mat);
		setMat4("world_to_view_matrix", view_mat);
		setMat4("inv_world_matrix", inv_world_mat);
		setMat4("model", model_mat);
		setVec3("direct_light", direct_light);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(3);

		glEnable(GL_DEPTH_TEST);
		//指定用于深度缓冲比较值；
		glDepthFunc(GL_LEQUAL);
		glDrawElements(GL_TRIANGLES, triangle_vertex_num, GL_UNSIGNED_INT, 0);

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	HANDLE_ERROR(cudaFree(dev_adj_index));
	HANDLE_ERROR(cudaFree(dev_adj_array));
	HANDLE_ERROR(cudaFree(dev_src_temper));
	HANDLE_ERROR(cudaFree(dev_dst_temper));

	return 0;
}












// utility function for checking shader compilation/linking errors.
// ------------------------------------------------------------------------
void checkCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	} else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

void init_shader(const char* vertexPath, const char* fragmentPath, GLuint &ID) {
	std::string glsl_version = "#version 450\n";
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;
	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		vShaderFile.open(vertexPath);
		fShaderFile.open(fragmentPath);
		std::stringstream vShaderStream, fShaderStream;
		// read file's buffer contents into streams
		vShaderStream <<glsl_version;
		fShaderStream <<glsl_version;
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		// close file handlers
		vShaderFile.close();
		fShaderFile.close();
		// convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}
	const char* vShaderCode = vertexCode.c_str();
	const char * fShaderCode = fragmentCode.c_str();
	// 2. compile shaders
	unsigned int vertex, fragment;
	int success;
	char infoLog[512];
	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");
	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");
	// shader Program
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}



// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, delta_time);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, delta_time);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, delta_time);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, delta_time);
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}