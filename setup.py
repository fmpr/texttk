# -*- coding: utf-8 -*-
# Copyright (C) 2016, Filipe Rodrigues <fmpr@dei.uc.pt>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import division
try:
    import setuptools
except:
    print '''
setuptools not found.

On linux, the package is often called python-setuptools'''
    from sys import exit
    exit(1)
long_description = file('README.rst').read()

packages = filter(lambda p: p.startswith('texttk'), setuptools.find_packages())
print packages

setuptools.setup(name = 'texttk',
      version = '0.1',
      description = 'Text Preprocessing ToolKit',
      long_description = long_description,
      author = u'Filipe Rodrigues',
      author_email = 'fmpr@dei.uc.pt',
      url = 'http://amilab.dei.uc.pt/fmpr/',
      license = 'Academic Free License',
      install_requires = ['nltk','sklearn','HTMLParser'],
      packages = packages,
      )


