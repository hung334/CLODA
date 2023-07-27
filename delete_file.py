import os



file_path = 'result_V4'
list_file = os.listdir(file_path)
#os.system('cd {}'.format(file_path))
os.chdir(file_path)


print(os.path.dirname(os.path.realpath(__file__)))


for name in list_file:
    #os.system('cd {}'.format(name))
    os.chdir(name)
    #print(os.path.dirname(os.path.realpath(__file__)))
    val = os.system('rm -r cam_mask')
    if (val):
        print(os.path.dirname(os.path.realpath(__file__)))
    #os.system('cd ..')
    os.chdir('../')

    
print('刪除完成')