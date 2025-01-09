build:
	docker-compose build

start:
	docker-compose up

restart-frontend:
	docker-compose restart frontend

restart-backend:
	docker-compose restart backend

logs-backend:
	docker-compose logs -f backend
