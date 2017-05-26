# end-to-end-for-plate-recognition
Python 3.5(64bit) 安装说明 http://www.cnblogs.com/ibyte/p/5871941.html

多标签分类,端到端的中文车牌识别基于mxnet .
从[xlvector的ocr代码](https://github.com/szad670401/learning-dl/tree/master/mxnet/ocr)修改，减少了参数，由于我没有显卡。单线程 9 samples/s 速度 ，用CPU在MBP上跑了50w张样本。识别率到了81%。不过还没有完全收敛。

##依赖:
 + Numpy
 + Mxnet
 + Opencv
 
##生成的车牌样张
通过渲染车牌加上畸变、噪声、与自然环境结合生成车牌的样本。

 ![image](plates_to_test/00.jpg)
  ![image](plates_to_test/01.jpg)
   ![image](plates_to_test/02.jpg)
    ![image](plates_to_test/03.jpg)
     ![image](plates_to_test/04.jpg)
        ![image](plates_to_test/06.jpg)
    ![image](plates_to_test/07.jpg)
     ![image](plates_to_test/08.jpg)   ![image](plates_to_test/02.jpg)
    ![image](plates_to_test/09.jpg)
     ![image](plates_to_test/10.jpg)
         ![image](plates_to_test/11.jpg)
##识别样张
 ![image](plates_to_test/Screen Shot 2016-08-07 at 12.51.56 AM.png)
 ![image](plates_to_test/Screen Shot 2016-08-07 at 12.53.41 AM.png)
  ![image](plates_to_test/Screen Shot 2016-08-07 at 12.55.45 AM.png)
 
