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
	int _numVertices;     //���ж�����������

	//size a: trangle verts; size b: origin verts
	std::vector<glm::vec3> _vertices;			// a     //���ж��������һ�������������(x,y,z)
	std::vector<glm::vec2> _texCoords;			// a     //�������꣨u��v��
	std::vector<glm::vec3> _normalVecs;			// a     //����
	std::vector<std::vector<int>> _adj_mat;		// b
	std::vector<glm::vec3> _origin_vertices;	// b
	std::vector<int> _vert_indexes;				// a	 // ��verticesӳ�䵽origin_vertices��
public:
	ImportedModel();
	ImportedModel(const char* filePath);
	int getNumVertices();
	std::vector<glm::vec3> getVertices();
	std::vector<glm::vec2> getTextureCoords();
	std::vector<glm::vec3> getNormals();

	std::vector<std::vector<int>> getAdjMat();
	std::vector<glm::vec3> getOriginVertices();
	std::vector<int> getVertIndexes();
};


class ModelImporter
{
private:
	std::vector<float> _vertVals;
	std::vector<float> _triangleVerts;
	std::vector<float> _textureCoords;
	std::vector<float> _stVals;
	std::vector<float> _normals;
	std::vector<float> _normVals;

	std::vector<int> _vertIndexes;
public:
	ModelImporter();
	void parseOBJ(const char* filePath, std::map<int, std::set<int>>& adj_mat);
	int getNumVertices();
	std::vector<float> getVertices();
	std::vector<float> getTextureCoordinates();
	std::vector<float> getNormals();
	std::vector<float> getOriginVertices();
	std::vector<int> getVertIdexes();

};


