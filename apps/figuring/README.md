
利用平面上上的gaze point数据逆映射到眼球体表面
眼球建模为直径24mm的球体
眼球到平面距离为70cm，显示器的尺度是 x,y \in (0,2000) * (0,1000)像素，对应的是27寸的显示器

## 创建
3D眼动数据可视化图，展示被试者的凝视点在三维空间中的分布情况
所有凝视点形成一个半球形点云结构, 3D立体视角，显示半球形结构

使用数据来自data/raw/all/all_trials_model_predictions_0111.csv中的original gaze point