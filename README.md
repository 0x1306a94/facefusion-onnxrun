本套程序是对最近火热的换脸程序 https://github.com/facefusion/facefusion 
的剥离。在官方程序里，定义了太多的.py文件和函数模块，模块之间的嵌套太复杂，
因此我重新编写了Python程序，我的程序里只有7个.py文件，换脸程序里一共包含5个模块，
除去main.py和utils.py文件，在我的程序里每个模块对应一个.py文件，接着我编写了C++程序。

onnx文件在百度云盘，链接：https://pan.baidu.com/s/12Fw2lqkhxWD5Xbk5A2Q2YQ 
提取码：sz78

### 下载模型文件
```bash
./script/download_weights.sh
```

### macOS 运行Python版本
```bash
#首先安装依赖
pip3 install -r python/requirements.txt --break-system-packages
# 运行
python3 python/main.py images/1.jpg images/target.jpg
```
### macOS 运行C++版本
```bash
#首先安装依赖
brew install opencv onnxruntime --verbose

mkdir build
cd build
cmake ../
cmake --build . --config Release
cmake --install . --config Release
./install/bin/facefusion-onnxrun --weights ./install/weights --out ./sample_out --source ../images/1.jpg --target ../images/target.jpg
```