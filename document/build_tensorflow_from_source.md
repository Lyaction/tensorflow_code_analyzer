# tensorflow 编译记录
  
build tensorflow from source.  
记录 mac osx 环境下构建 tensorflow 源码的过程  
  
## 准备工作
  
1、下载 tensorflow1.12 源码  
  
`git clone https://github.com/tensorflow/tensorflow.git`  
  
2、安装 python 环境  
  
`conda create env=$name python=2.7`  
`conda install tensorflow=1.12`  
  
3、安装 bazel 0.14.1  
  
`wget https://github.com/bazelbuild/bazel/releases/download/0.14.1/bazel-0.14.1-installer-darwin-x86_64.shi`  
`chmod +x bazel-0.14.1-installer-darwin-x86_64.sh`  
`sh bazel-0.14.1-installer-darwin-x86_64.sh`  
  
4、修改 configure.py  
  
`check_bazel_version('0.15.0') -> check_bazel_version('0.14.0')`  
  
5、修改 WORKSPACE  
  
`check_bazel_version_at_least("0.15.0") -> check_bazel_version_at_least("0.14.0")`  
  
## 编译&安装

1、编译  

保证编译的最小依赖，全部填 `n`  

`./configure`  
`bazel build --verbose_failures --config=opt //tensorflow/tools/pip_package:build_pip_package`    
  
2、生成 whl
  
`bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`  
  
生成文件 `eg:/tmp/tensorflow_pkg/tensorflow-1.12.3-cp27-cp27m-macosx_10_16_x86_64.whl`  
  
3、安装 tensorflow1.12  
  
安装前 check 环境 `python2.7 and mac version: 10.16 x86.64`  
  
`python -m pip install /tmp/tensorflow_pkg/tensorflow-1.12.3-cp27-cp27m-macosx_10_16_x86_64.whl`  
  
安装所有缺失的依赖 `pip install XXX or conda install XXX`  
  
4、测试  
  
`cd ~; python -c "import tensorflow as tf; hello = tf.constant('Hello, TensorFlow!'); sess = tf.Session(); print(sess.run(hello))"`  
  
`If success, Congratulations!`  
