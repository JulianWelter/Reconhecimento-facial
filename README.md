# Docker DB
docker network create --driver bridge postgres-network
docker run --name teste-postgres --network=postgres-network -e "POSTGRES_PASSWORD=root" -p 5432:5432 -v /home/renatogroffe/Desenvolvimento/PostgreSQL:/var/lib/postgresql/data -d postgres
docker run --name teste-pgadmin --network=postgres-network -p 15432:80 -e "PGADMIN_DEFAULT_EMAIL=email@mail.com" -e "PGADMIN_DEFAULT_PASSWORD=root!" -d dpage/pgadmin4

# Docker project
docker build -t myimage .
docker run --network=postgres-network -d --name mycontainer -p 80:80 myimage
