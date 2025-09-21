from PIL import Image
import os

def compress_image(input_path, output_path, max_size_kb=200, step=5):
    """
    自動壓縮 JPG 圖片到指定大小 (KB) 以下，保持解析度不變
    input_path : 原始圖片路徑
    output_path: 輸出圖片路徑
    max_size_kb: 最大檔案大小 (KB)
    step       : 每次降低品質的幅度 (例如 5 表示從95 -> 90 -> 85...)
    """
    # 開啟圖片
    img = Image.open(input_path)

    # 確保是 RGB 模式（JPG 不支援透明度）
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 95  # 起始畫質
    while quality > 5:
        # 儲存暫存檔
        img.save(output_path, "JPEG", quality=quality, optimize=True)
        size_kb = os.path.getsize(output_path) / 1024

        print(f"嘗試品質 {quality} -> 檔案大小 {size_kb:.1f} KB")

        if size_kb <= max_size_kb:
            print(f"✅ 壓縮完成，檔案大小 {size_kb:.1f} KB，品質 {quality}")
            return

        quality -= step

    print("⚠ 無法在合理品質下壓縮到指定大小，請考慮降低解析度")

# === 使用範例 ===
compress_image("input.jpg", "output.jpg", max_size_kb=200)
