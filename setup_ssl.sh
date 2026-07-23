#!/bin/bash

echo "🛑 Menghentikan server uvicorn yang sedang berjalan di port 80..."
fuser -k 80/tcp

echo "📦 Menginstal Nginx dan Certbot..."
apt-get update
apt-get install -y nginx certbot python3-certbot-nginx

echo "⚙️ Mengkonfigurasi Nginx untuk Reverse Proxy..."
cat > /etc/nginx/sites-available/vps.temaniskripsi.id << 'EOF'
server {
    listen 80;
    server_name vps.temaniskripsi.id;

    # Agar bisa upload file 100MB
    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Mencegah Nginx memotong koneksi yang lama (Timeout 1 jam)
        proxy_read_timeout 3600s;
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        
        # Disable buffering untuk stream
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }
}
EOF

ln -sf /etc/nginx/sites-available/vps.temaniskripsi.id /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

echo "🔐 Mendapatkan Sertifikat SSL dari Let's Encrypt..."
certbot --nginx -d vps.temaniskripsi.id --non-interactive --agree-tos -m admin@temaniskripsi.id

echo "🔄 Merestart Nginx..."
systemctl restart nginx

echo "✅ SSL Berhasil dipasang! VPS sekarang bisa diakses melalui HTTPS!"
