#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;

class ImportedModel
{
private:
	int _numVertices;     //所有顶点坐标总数

	//size a: trangle verts; size b: origin verts
	std::vector<int> _triangle_indexes;			// a     //所有三角形顶点索引
	std::vector<glm::vec2> _texCoords;			// b     //纹理坐标（u，v）
	std::vector<glm::vec3> _normalVecs;			// b     //法线
	std::vector<std::vector<int>> _adj_mat;		// b
	std::vector<glm::vec3> _origin_vertices;	// b
public:
	ImportedModel();
	ImportedModel(const char* filePath);
	int getNumVertices();
	std::vector<int> getTriangleIndexes();
	std::vector<glm::vec2> getTextureCoords();
	std::vector<glm::vec3> getNormals();

	std::vector<std::vector<int>> getAdjMat();
	std::vector<glm::vec3> getOriginVertices();
};


class ModelImporter
{
private:
	std::vector<float> _vertVals;
	std::vector<int> _triangleVerts;
	std::vector<float> _textureCoords;
	std::vector<float> _stVals;
	std::vector<float> _normals;
	std::vector<float> _normVals;
public:
	ModelImporter();
	void parseOBJ(const char* filePath, std::map<int, std::set<int>>& adj_mat);
	int getNumVertices();
	std::vector<int> getTriangleVertices();
	std::vector<float> getTextureCoordinates();
	std::vector<float> getNormals();
	std::vector<float> getOriginVertices();

};


