# sherpa-ncnn 部署

## IOS部署

在本节中，我们将介绍如何构建用于语音的 iOS 应用程序 使用 sherpa-ncnn 进行识别并在 Mac 上的模拟器中运行它， 在您的 iPhone 或 iPad 上运行它。real-time

我们还提供实时语音识别的视频演示。

提示

在语音识别过程中，它不需要访问 Internet。 每个 Bug 都在您的 iPhone 或 iPad 上本地处理。

视频演示
视频 1：iPhone 14 Pro 中文 + 英文（模拟器）
视频 2：iPad 11 Pro 上的中文 + 英文（模拟器）
构建适用于 iOS 的 sherpa-ncnn
要求
下载 sherpa-ncnn
构建 sherpa-ncnn（在命令行中，C++ 部分）
构建 sherpa-ncnn（在 Xcode 中）
在 iPhone/iPad 上运行 sherpa-ncnn
对于更好奇的
运行 ./build-ios.sh 生成的文件
openmp.xc框架
sherpa-ncnn.xc框架
如何在 Xcode 中使用 ./build-ios.sh 生成的文件