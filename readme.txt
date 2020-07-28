1. main.py为主程序。语言为python，需要numpy、opencv、PyQt等库。

2. camera_configs.py、Calibrateresult.yml为相机参数相关文件。

3. GUI文件夹内存放GUI代码文件。main.py里from GUI import ...即是这个文件夹

4. 测试视频和图片存放用来测试的视频和双目图像。

5. GUI交互界面（程序运行界面）中，读取双目图像、计算深度图、计算最佳降落位置、清空显示，依次点击可以读取并测试单张的图像。动态显示可以读取测试一段视频。block_size、num分别是计算深度图时的匹配参数，按预设即可。

6. exe文件为打包成的PC程序，可能会因电脑环境等原因运行失败。如果无法成功运行，请运行main.py文件。