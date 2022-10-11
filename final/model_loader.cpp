#include "model_loader.h"
#include <sstream>


ImportedModel::ImportedModel()
{

}

ImportedModel::ImportedModel(const char* filePath)
{
	ModelImporter modelImporter = ModelImporter();

	std::map<int, std::set<int>> adj_init;
	modelImporter.parseOBJ(filePath, adj_init);

	_numVertices = modelImporter.getNumVertices();

	vector<float> origin_verts = modelImporter.getOriginVertices();
	vector<float> verts = modelImporter.getVertices();
	vector<float> tcs = modelImporter.getTextureCoordinates();
	vector<float> normals = modelImporter.getNormals();
	_vert_indexes = modelImporter.getVertIdexes();

	for (int i = 0; i < _numVertices; i++)
	{
		_vertices.push_back(glm::vec3(verts[i * 3 + 0], verts[i * 3 + 1], verts[i * 3 + 2]));
		_texCoords.push_back(glm::vec2(tcs[i * 2 + 0], tcs[i * 2 + 1]));
		_normalVecs.push_back(glm::vec3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]));
	}

	int num_origin = origin_verts.size() / 3;
	for (int i = 0; i < num_origin; i++) {
		_origin_vertices.push_back(glm::vec3(origin_verts[i * 3 + 0], origin_verts[i * 3 + 1], origin_verts[i * 3 + 2]));
	}
	_adj_mat.resize(num_origin, vector<int>());
	for (auto& iter: adj_init) {
		int i = iter.first;
		for (auto neighbour: iter.second) {
			_adj_mat[i].push_back(neighbour);
		}
	}
}

int ImportedModel::getNumVertices()
{
	return std::move(_numVertices);
}

std::vector<glm::vec3> ImportedModel::getVertices()
{
	return std::move(_vertices);
}

std::vector<glm::vec2> ImportedModel::getTextureCoords()
{
	return std::move(_texCoords);
}

std::vector<glm::vec3> ImportedModel::getNormals()
{
	return std::move(_normalVecs);
}

std::vector<std::vector<int>> ImportedModel::getAdjMat() 
{
	return std::move(_adj_mat);
}

std::vector<glm::vec3> ImportedModel::getOriginVertices() 
{
	return std::move(_origin_vertices);
}

std::vector<int> ImportedModel::getVertIndexes() 
{
	return _vert_indexes;
}

/// <summary>
/// ModelImporter implement
/// </summary>

ModelImporter::ModelImporter()
{

}

void ModelImporter::parseOBJ(const char* filePath, std::map<int, std::set<int>>& adj_mat)
{
	float x = 0.f, y = 0.f, z = 0.f;
	string content;
	ifstream fileStream(filePath, ios::in);
	string line = "";

	while (!fileStream.eof())
	{
		getline(fileStream, line);
		if (line.compare(0, 2, "v ") == 0)    //注意v后面有空格
		{
			std::stringstream ss(line.erase(0, 1));
			ss >> x >> y >> z;
			//ss >> x; ss >> y; ss >> z;
			_vertVals.push_back(x);
			_vertVals.push_back(y);
			_vertVals.push_back(z);
		}
		if (line.compare(0, 2, "vt") == 0)
		{
			std::stringstream ss(line.erase(0, 2));
			ss >> x >> y;
			_stVals.push_back(x);
			_stVals.push_back(y);
		}
		if (line.compare(0, 2, "vn") == 0)
		{
			std::stringstream ss(line.erase(0, 2));
			ss >> x >> y >> z;
			_normVals.push_back(x);
			_normVals.push_back(y);
			_normVals.push_back(z);
		}
		if (line.compare(0, 1, "f") == 0)  //原书有误
		{
			string oneCorner, v, t, n;
			std::stringstream ss(line.erase(0, 2));
			int idx[3];
			for (int i = 0; i < 3; i++)
			{
				getline(ss, oneCorner, ' ');
				//getline(ss, oneCorner, " ");
				stringstream oneCornerSS(oneCorner);
				getline(oneCornerSS, v, '/');
				getline(oneCornerSS, t, '/');
				getline(oneCornerSS, n, '/');

				idx[i] = stoi(v) - 1;

				_vertIndexes.push_back(idx[i]);

				int vertRef = idx[i] * 3;   //为什么要 -1？
				int tcRef = (stoi(t) - 1) * 2;
				int normRef = (stoi(n) - 1) * 3;

				_triangleVerts.push_back(_vertVals[vertRef]);
				_triangleVerts.push_back(_vertVals[vertRef + 1]);
				_triangleVerts.push_back(_vertVals[vertRef + 2]);

				_textureCoords.push_back(_stVals[tcRef]);
				_textureCoords.push_back(_stVals[tcRef + 1]);

				_normals.push_back(_normVals[normRef]);
				_normals.push_back(_normVals[normRef + 1]);
				_normals.push_back(_normVals[normRef + 2]);
			}

			//这里adj插入的是_vertVals之间的关系，而不是_triangleVerts之间的关系！！！ todo
			// 应该给cuda origin vertexes和colors以及对应的indexes
			adj_mat[idx[0]].insert(idx[1]);
			adj_mat[idx[0]].insert(idx[2]);
			adj_mat[idx[1]].insert(idx[0]);
			adj_mat[idx[1]].insert(idx[2]);
			adj_mat[idx[2]].insert(idx[0]);
			adj_mat[idx[2]].insert(idx[1]);
		}
	}
}

int ModelImporter::getNumVertices()
{
	return (_triangleVerts.size() / 3);
}

std::vector<float> ModelImporter::getVertices()
{
	return std::move(_triangleVerts);
}

std::vector<float> ModelImporter::getTextureCoordinates()
{
	return std::move(_textureCoords);
}

std::vector<float> ModelImporter::getNormals()
{
	return std::move(_normals);
}

std::vector<float> ModelImporter::getOriginVertices() 
{
	return std::move(_vertVals);
}

std::vector<int> ModelImporter::getVertIdexes()
{
	return std::move(_vertIndexes);
}