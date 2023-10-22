<img src="https://github.com/DarkSleeper/Cuda-Heat-Spread/assets/48831197/10c92701-9e89-44cb-8328-9fde635dff22" width="640">

热量扩散具体方法<br>
为了实现GPU并行计算的热量扩散，首先应搭建好CPU控制的模型文件读入、数据处理、实时计算与绘制的整体框架。

下面重点介绍CUDA热量扩散程序以及邻接矩阵的处理。<br>
在热量扩散的计算中，本实验利用了OBJ模型文件的三角面描述，构建了顶点之间的连接关系。使用提供的简化热传递模型：假设热源恒温不变，模型的温度没有自然耗散，每一帧之间的时间跨度很小。<br>
那么可以近似认为每一个节点的温度变化满足以下条件：<br>
	1. 该节点的下一帧温度，只跟相邻的节点相关。<br>
	2. 两个节点之间的热量传递速率和两个节点的温差dT成正比。<br>
	3. 两个节点之间的热量传递速率和两个节点之间的距离dS成反比。<br>
	4.忽略材质因素的影响（如热导率、比热容、和密度）。

在获得当前帧的顶点温度后，为了将其可视化，我们还需要将温度值转化为颜色进行存储。定义绿色(0, 1, 0, 1)为最低温度，红色(1, 0, 0, 1)为最高温度，那么温度转化为颜色的kernel程序核心代码如下：<br>
```c++
float a = (temper[index] - MIN_TEMPER) / (MAX_TEMPER - MIN_TEMPER);
if (a > 0.5) {
	color[index].x = 1;
	color[index].y = (1 - a) / a;
} else {
	color[index].x = a / (1 - a);
	color[index].y = 1;
}
color[index].z = 0.f;
color[index].w = 1.f;
```

这里不使用单一的线性插值，是为了避免顶点颜色在中间温度时，导致r值与g值都偏小，进而使显示的颜色不明显，过渡效果不宜观察。将颜色中最大的分量始终保持为1，可以更好地体现热量的传播。<br>
在上面用到的数据中，颜色与位置存储在VBO中，是线性数组，并通过递增的顶点索引访问。由于温度需要在每帧进行变化，因此也是同种类型的线性数组，并且在CUDA Memory中部署两份，一份作为只读的当前温度，一份作为只写的下一帧温度，并在每帧结束时进行指针的交换。<br>
余下的邻接矩阵A和顶点位置数组P，在CPU中用std::map存储。为了更好地进行并行计算，本实验设计了两种方案，分别将其转化为CSR格式与ELL格式进行存储，在部署到CUDA Memory中。转化的代码如下所示：<br>
转化为CSR格式：<br>
```c++
	vector<int> adj_index(vertex_num + 1, 0);
	vector<int>	adj_array;
	vector<float> dis_array;
	for (int i = 0; i < vertex_num; i++) {
		adj_index[i] = adj_array.size();
		float3 src_pos = make_float3(pValues[i*3], pValues[i*3 + 1], pValues[i*3 + 2]);
		for (auto neighbour: adj_map[i]) {
			adj_array.push_back(neighbour);
			float3 dst_pos = make_float3(pValues[neighbour*3], pValues[neighbour*3 + 1], pValues[neighbour*3 + 2]);
			auto distance = get_distance(src_pos, dst_pos);
			dis_array.push_back(distance);
		}
	}
```

转化为ELL格式：<br>
```c++
	vertex_num_aligned = ((vertex_num + THREAD_NUM - 1) / THREAD_NUM) * THREAD_NUM;
	vector<int> adj_ell_array(vertex_num_aligned * max_adj_num, -1);
	vector<float> dis_array(vertex_num_aligned * max_adj_num, -1.f);
	for (int i = 0; i < vertex_num; i++) {
		int adj_num = adj_map[i].size();
		float3 src_pos = make_float3(pValues[i*3], pValues[i*3 + 1], pValues[i*3 + 2]);
		for (int j = 0; j < adj_num; j++) {
			auto neighbour = adj_map[i][j];
			adj_ell_array[j * vertex_num_aligned + i] = neighbour;
			float3 dst_pos = make_float3(pValues[neighbour*3], pValues[neighbour*3 + 1], pValues[neighbour*3 + 2]);
			dis_array[j*vertex_num_aligned + i] = get_distance(src_pos, dst_pos);
		}
	}
```
这里的处理构造了CSR与ELL格式的邻接矩阵，描述了各顶点之间的连结关系，以及两者之间距离的初始值。为了避免模型在运行时改变产生的影响，还应该在每帧的“CUDA热量扩散程序”模块中用kernel程序进行距离的重新计算与存储。<br>
