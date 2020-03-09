from setuptools import setup, find_packages

package_data = {

}

setup(name='tf_object_detection_util',
      version='0.1.8',
      description='A library to make Object Detection using Tensorflow Object Detection API easier.',
      url='https://github.com/prasannals/tf_object_detection_util',
      author='Prasanna Lakkur Subramanyam',
      author_email='prasanna.lakkur@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
            'pandas',
            'numpy',
            'six',
            'tensorflow-gpu==1.9.0',
            'matplotlib',
            'opencv_python>=4.1.2.30',
            'Pillow>=7.0.0'
      ],
      zip_safe=False)