docker build -t "mo640" .
#docker run -it -v ~/git/mo640:/external/mo640 --name mo640 mo640
docker run -it --name mo640 mo640

# kill all containers
# docker rm $(docker ps -a -q)

# kill all images
# docker rmi $(docker images -q)