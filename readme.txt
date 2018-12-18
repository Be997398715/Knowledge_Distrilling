相关视频及资料：https://www.bilibili.com/video/av21152136/
		https://github.com/dkozlov/awesome-knowledge-distillation#papers
		https://blog.csdn.net/qzrdypbuqk/article/details/81482598

论文中提出的方法：使用已训练好的参数多精度高的网络作为 "老师" 对参数少网络小精度低的网络 "学生" 进行知识的蒸馏，提高小网络的精度
本次实验中使用了自己已经训练好的 residual_attention 网络作为老师，使用了 hard_logits 和 soft_logits 的整合作为 loss ，最终将
squeeze_net 在 Cifar-10 数据集上的精度提高了 5% ，取得了较行的结果，而与 residual_attention 网络比较参数从 128M 缩小到了 3M 。
论文中的内容较为易懂，可以参考上面的资料进行解读。

训练： python train_knowledge_distilling.py (训练和测试蒸馏后的模型)
训练： python train_student.py (训练和测试学生网络--squeeze_net)
## 测试： python predict_and_visuable.py(TTA和特征可视化)  "没有进行可视化和TTA，仅做实验"
模型相关：models文件夹下
测试图片：test_images文件夹下
保存模型权重：logs文件夹下


注意： 小网络一定要选择好，不然再好的蒸馏也不一定会令人满意，因为小网络参数少表达能力弱。如果这次我选择类型resnet的精简小网络
       那么相信最终的结果会更好。不过经过实验和查找资料证明 蒸馏 只是一种提高网络精度的方法，可以看作模型的增强--emssemble，
       压缩模型只是附带的一个结果。