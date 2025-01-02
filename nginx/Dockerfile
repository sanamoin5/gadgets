FROM nginx:alpine
LABEL maintainer="sanamoin"

COPY nginx.conf /etc/nginx/conf.d/default.conf

# The build folder will be mounted at runtime
WORKDIR /usr/share/nginx/html
