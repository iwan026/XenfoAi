#!/data/data/com.termux/files/usr/bin/bash

# Script untuk auto push GitHub di Termux dengan token API
# Simpan script ini dengan nama gitpush.sh
# Berikan izin eksekusi: chmod +x gitpush.sh
# Gunakan: ./gitpush.sh "Pesan commit Anda"

# Warna untuk output
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

# Lokasi file konfigurasi token
CONFIG_DIR="$HOME/.git-credentials"
TOKEN_FILE="$HOME/.git-token"

# Fungsi untuk menampilkan banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════╗"
    echo "║      AUTO GITHUB PUSH - TERMUX        ║"
    echo "║         WITH API TOKEN SUPPORT        ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

# Fungsi untuk memeriksa apakah git terinstall
check_git() {
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Git belum terinstall!${NC}"
        echo -e "${YELLOW}Menginstall git...${NC}"
        pkg update -y && pkg install git -y
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Gagal menginstall git. Silakan install manual dengan perintah:${NC}"
            echo -e "${CYAN}pkg update -y && pkg install git -y${NC}"
            exit 1
        else
            echo -e "${GREEN}Git berhasil diinstall!${NC}"
        fi
    fi
}

# Fungsi untuk memeriksa apakah folder saat ini adalah repo git
check_git_repo() {
    if [ ! -d .git ]; then
        echo -e "${RED}Folder saat ini bukan repository git!${NC}"
        echo -e "${YELLOW}Apakah Anda ingin menginisialisasi git? (y/n)${NC}"
        read -r answer
        
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            git init
            echo -e "${GREEN}Git repository berhasil diinisialisasi!${NC}"
        else
            echo -e "${RED}Batal. Script dihentikan.${NC}"
            exit 1
        fi
    fi
}

# Fungsi untuk memeriksa dan mengatur konfigurasi git jika belum ada
check_git_config() {
    if [ -z "$(git config --global user.name)" ] || [ -z "$(git config --global user.email)" ]; then
        echo -e "${YELLOW}Konfigurasi Git belum lengkap. Mari kita atur:${NC}"
        
        echo -e "${CYAN}Masukkan nama pengguna GitHub Anda:${NC}"
        read -r git_name
        git config --global user.name "$git_name"
        
        echo -e "${CYAN}Masukkan email GitHub Anda:${NC}"
        read -r git_email
        git config --global user.email "$git_email"
        
        echo -e "${GREEN}Konfigurasi Git berhasil disimpan!${NC}"
    fi
    
    # Konfigurasi credential helper
    if [ -z "$(git config --global credential.helper)" ]; then
        git config --global credential.helper store
        echo -e "${GREEN}Credential helper diaktifkan untuk menyimpan token.${NC}"
    fi
}

# Fungsi untuk mengatur token GitHub
setup_github_token() {
    # Cek apakah token sudah ada
    if [ -f "$TOKEN_FILE" ]; then
        token=$(cat "$TOKEN_FILE")
        username=$(git config --global user.name)
        
        # Cek apakah kredensial sudah terkonfigurasi
        if ! grep -q "https://" "$CONFIG_DIR" 2>/dev/null; then
            echo -e "${YELLOW}Mengkonfigurasi kredensial GitHub...${NC}"
            echo "https://$username:$token@github.com" > "$CONFIG_DIR"
            chmod 600 "$CONFIG_DIR"
            echo -e "${GREEN}Kredensial berhasil dikonfigurasi!${NC}"
        fi
        
        return 0
    fi
    
    echo -e "${YELLOW}Token GitHub belum dikonfigurasi.${NC}"
    echo -e "${CYAN}Untuk push otomatis tanpa memasukkan kredensial, Anda perlu Personal Access Token.${NC}"
    echo -e "${CYAN}1. Buka https://github.com/settings/tokens${NC}"
    echo -e "${CYAN}2. Klik 'Generate new token' (Classic)${NC}"
    echo -e "${CYAN}3. Beri nama token dan centang 'repo' untuk akses penuh ke repositori${NC}"
    echo -e "${CYAN}4. Scroll ke bawah dan klik 'Generate token'${NC}"
    echo -e "${CYAN}5. Salin token yang dihasilkan dan tempel di sini${NC}"
    echo -e "${CYAN}Masukkan Personal Access Token GitHub Anda:${NC}"
    read -rs token
    
    if [ -z "$token" ]; then
        echo -e "${RED}Token tidak boleh kosong!${NC}"
        return 1
    fi
    
    # Simpan token ke file dengan izin yang aman
    echo "$token" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    
    # Tambahkan kredensial ke git credential store
    username=$(git config --global user.name)
    echo "https://$username:$token@github.com" > "$CONFIG_DIR"
    chmod 600 "$CONFIG_DIR"
    
    echo -e "${GREEN}Token GitHub berhasil disimpan! Tidak perlu memasukkan username/password lagi.${NC}"
    return 0
}

# Fungsi untuk memeriksa dan menambahkan remote jika belum ada
check_remote() {
    if ! git remote -v | grep -q origin; then
        echo -e "${YELLOW}Remote repository belum dikonfigurasi.${NC}"
        echo -e "${CYAN}Masukkan URL repository GitHub Anda:${NC}"
        echo -e "${CYAN}(contoh: https://github.com/username/repo.git)${NC}"
        read -r repo_url
        
        # Jika URL menggunakan https://, konversi untuk menggunakan token
        if [[ "$repo_url" == https://* ]] && [ -f "$TOKEN_FILE" ]; then
            username=$(git config --global user.name)
            token=$(cat "$TOKEN_FILE")
            
            # Ekstrak bagian URL setelah github.com
            repo_path=$(echo "$repo_url" | sed 's|https://github.com/||')
            
            # Buat URL dengan kredensial
            auth_url="https://$username:$token@github.com/$repo_path"
            git remote add origin "$auth_url"
        else
            git remote add origin "$repo_url"
        fi
        
        echo -e "${GREEN}Remote repository berhasil ditambahkan!${NC}"
    fi
}

# Fungsi utama untuk push ke GitHub
push_to_github() {
    # Mendapatkan pesan commit dari argumen atau meminta input
    if [ -z "$1" ]; then
        echo -e "${CYAN}Masukkan pesan commit:${NC}"
        read -r commit_message
    else
        commit_message="$1"
    fi
    
    echo -e "${YELLOW}Memulai proses push ke GitHub...${NC}"
    
    # Menambahkan semua perubahan
    echo -e "${CYAN}[1/4] Menambahkan semua file ke staging area...${NC}"
    git add .
    
    # Melakukan commit
    echo -e "${CYAN}[2/4] Melakukan commit dengan pesan: ${commit_message}${NC}"
    git commit -m "$commit_message"
    
    # Mengambil perubahan terbaru dari remote (jika ada)
    echo -e "${CYAN}[3/4] Mengambil perubahan terbaru dari remote...${NC}"
    git pull origin "$(git rev-parse --abbrev-ref HEAD)" --no-edit
    
    # Melakukan push dengan protokol https yang sudah terotentikasi
    echo -e "${CYAN}[4/4] Melakukan push ke GitHub...${NC}"
    git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
    
    # Menampilkan status akhir
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Push ke GitHub berhasil!${NC}"
    else
        echo -e "${RED}Terjadi kesalahan saat push ke GitHub.${NC}"
        echo -e "${YELLOW}Silakan periksa pesan error di atas.${NC}"
    fi
}

# Memulai eksekusi script
show_banner
check_git
check_git_repo
check_git_config
setup_github_token
check_remote
push_to_github "$1"

echo -e "${GREEN}Proses selesai! Terima kasih telah menggunakan script ini.${NC}"