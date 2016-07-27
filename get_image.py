# -*- coding: utf-8 -*-

import requests, os, time

headers = {'User-Agent' : 'Mozilla/5.0'}
result_dir = '/tmp/download/'
list_file = '/tmp/list.txt'
done_file = '/tmp/done.txt'
fail_file = '/tmp/fail.txt'

def fetchImage(data):
    label = data[0]
    url = data[1]
    path_relative = url.replace('http://', '').replace('https://', '')
    try:
        res = requests.get(url)
        image = res.content
        path = result_dir + label + '/'
        if not os.path.exists(path):
            os.mkdir(path)
        file = url.rsplit('/', 1)[1].split('?')[0]
        with open('{path}{file}'.format(path = path, file = file), 'wb') as f:
            f.write(image)
    except:
        return False
    return True

def getTarget():
    result = []
    with open(list_file, 'r') as f:
        url_list = f.read().split('\n')
    result = url_list.pop(0).split(',')
    with open(list_file, 'w') as f:
        f.write('\n'.join(url_list))
    return result

def saveUrl(file_name, url):
    with open(file_name, 'a') as f:
        f.write(url + '\n')

def download():
    data = getTarget()
    while data != ['']:
        if fetchImage(data):
            saveUrl(done_file, data[1])
            print('done ' + data[1])
        else:
            saveUrl(fail_file, data[1])
            print('fail ' + data[1])
        data = getTarget()
        time.sleep(0.5)

download()

