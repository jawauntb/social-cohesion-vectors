FROM nginx:1.27-alpine

COPY site/ /usr/share/nginx/html/
COPY deploy/railway-nginx.conf.template /etc/nginx/templates/default.conf.template
