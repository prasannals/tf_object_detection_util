from setuptools import setup, find_packages

package_data = {

}

setup(name='tf_object_detection_util',
      version='0.1.3',
      description='A library to make Object Detection using Tensorflow Object Detection API easier.',
      url='https://github.com/prasannals/tf_object_detection_util',
      author='Prasanna Lakkur Subramanyam',
      author_email='prasanna.lakkur@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)