# 温度采样

通过温度 T 调节 softmax 分布尖锐程度：
- T<1：更确定（sharpen）
- T>1：更多样（flatten）
- T→0：退化为贪心 argmax

详见 `温度采样.ipynb`。
