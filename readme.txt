�����Ƶ�����ϣ�https://www.bilibili.com/video/av21152136/
		https://github.com/dkozlov/awesome-knowledge-distillation#papers
		https://blog.csdn.net/qzrdypbuqk/article/details/81482598

����������ķ�����ʹ����ѵ���õĲ����ྫ�ȸߵ�������Ϊ "��ʦ" �Բ���������С���ȵ͵����� "ѧ��" ����֪ʶ���������С����ľ���
����ʵ����ʹ�����Լ��Ѿ�ѵ���õ� residual_attention ������Ϊ��ʦ��ʹ���� hard_logits �� soft_logits ��������Ϊ loss �����ս�
squeeze_net �� Cifar-10 ���ݼ��ϵľ�������� 5% ��ȡ���˽��еĽ�������� residual_attention ����Ƚϲ����� 128M ��С���� 3M ��
�����е����ݽ�Ϊ�׶������Բο���������Ͻ��н����

ѵ���� python train_knowledge_distilling.py (ѵ���Ͳ���������ģ��)
ѵ���� python train_student.py (ѵ���Ͳ���ѧ������--squeeze_net)
## ���ԣ� python predict_and_visuable.py(TTA���������ӻ�)  "û�н��п��ӻ���TTA������ʵ��"
ģ����أ�models�ļ�����
����ͼƬ��test_images�ļ�����
����ģ��Ȩ�أ�logs�ļ�����


ע�⣺ С����һ��Ҫѡ��ã���Ȼ�ٺõ�����Ҳ��һ�����������⣬��ΪС��������ٱ������������������ѡ������resnet�ľ���С����
       ��ô�������յĽ������á���������ʵ��Ͳ�������֤�� ���� ֻ��һ��������羫�ȵķ��������Կ���ģ�͵���ǿ--emssemble��
       ѹ��ģ��ֻ�Ǹ�����һ�������