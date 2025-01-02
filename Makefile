build:
	docker-compose build

start:
	docker-compose up

restart-frontend:
	docker-compose restart frontend

logs-backend:
	docker-compose logs -f backend
