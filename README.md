Download an image, load it up in opencv2.

```
curl -O "http://www.handsonbanking.org/financial-education/wp-content/uploads/2012/10/YA_01_05_05_en.jpg"
vagrant@vagrant-ubuntu-trusty-64:~$ mv YA_01_05_05_en.jpg check.png
vagrant@vagrant-ubuntu-trusty-64:~$ python
Python 2.7.6 (default, Mar 22 2014, 22:59:56) 
[GCC 4.8.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
libdc1394 error: Failed to initialize libdc1394
>>> img = cv2.imread("check.png")
>>> img
array([[[ 15, 180, 143],
        [ 25, 183, 147],
        [ 30, 173, 140],
        ..., 
        [ 17, 188, 149],
        [ 22, 186, 145],
        [ 26, 185, 145]],
```


