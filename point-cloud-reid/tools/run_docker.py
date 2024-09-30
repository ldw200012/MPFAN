import os 

# path_to_reid_from_pc = '/home/dongwooklee1201/morin/lidar_reid_ws_light' #path outside of docker 
path_to_reid_from_pc = '/mnt/0578ea01-61ee-40b7-84ae-4a44f210c570/Dongwook/lidar_reid_ws_light' #path outside of docker 
# path_to_reid_data = '/media/dongwooklee1201/0129b558-20d3-46bb-b9e9-82a19297be2e4/home/dongwooklee1201/Datasets'   #to make symbolic link to data work
path_to_reid_data = '/mnt/0578ea01-61ee-40b7-84ae-4a44f210c570/Dongwook/Datasets'   #to make symbolic link to data work

memory = '724g'
cpus = 10
gpus = 'all'
port = 14000
image = 'daldidan/dev:bentherien_modified'
name = 'reid_light'

command = "docker run -v {}:{} -v {}:{} --memory {} --shm-size=8g --cpus={} --gpus {} -p {}:{} --name {} --rm -it {}".format(
    path_to_reid_from_pc,"/home/bentherien/",path_to_reid_data,"/home/bentherien/point-cloud-reid/Datasets/",memory,cpus,gpus,port,port,name,image
)

print("################################################")
print('[run_docker.py executing] ',command)
print("################################################")

os.system(command)