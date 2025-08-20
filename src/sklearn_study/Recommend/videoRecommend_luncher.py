import mainModel as mm
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

def get_user_id_and_recommend():
    root = tk.Tk()
    root.title("短视频推荐系统")
    root.geometry("500x400")

    large_font = ("SimHei", 16)
    result_font = ("SimHei", 12)

    label = ttk.Label(root, text="请输入用户ID：", font=large_font)
    label.pack(pady=20)

    entry = ttk.Entry(root, font=large_font, width=15)
    entry.pack()

    error_label = ttk.Label(root, text="", font=("SimHei", 12), foreground="red")
    error_label.pack(pady=10)

    status_label = ttk.Label(root, text="", font=("SimHei", 12), foreground="blue")
    status_label.pack(pady=5)

    progress = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate")
    # 默认不显示进度条
    progress.pack_forget()

    result_frame = ttk.Frame(root)
    result_label = ttk.Label(result_frame, text="推荐结果将显示在这里...", font=result_font, wraplength=350, justify="left")
    result_label.pack(pady=10)

    def recommend_thread(user_id):
        try:
            # 显示进度条并重置为0
            progress.pack(pady=5)
            progress["value"] = 0
            status_label.config(text="加载数据库中...")
            root.update_idletasks()
            progress["value"] = 20
            time.sleep(0.5)

            status_label.config(text="正在根据视频数据库和视频质量为用户推荐优质视频...")
            root.update_idletasks()
            progress["value"] = 60

            recommendations = mm.main(user_id)

            status_label.config(text="计算完成！")
            progress["value"] = 100
            root.update_idletasks()
            time.sleep(0.5)  # 显示满格一会儿

            # 兼容字典或列表
            if isinstance(recommendations, dict) and recommendations:
                result_text = f"为用户 {user_id} 推荐的视频：\n"
                for tag, vid in recommendations.items():
                    result_text += f"类别 {tag} 推荐视频ID: {vid}\n"
                result_label.config(text=result_text, foreground="black")
            elif isinstance(recommendations, list) and recommendations:
                result_text = f"为用户 {user_id} 推荐的视频：\n" + "\n".join([f"- {item}" for item in recommendations[:5]])
                result_label.config(text=result_text, foreground="black")
            else:
                result_label.config(text="未找到推荐结果", foreground="orange")
        except Exception as e:
            result_label.config(text="未找到推荐结果", foreground="orange")
            messagebox.showerror("错误", f"推荐过程出错：{str(e)}")
        finally:
            # 计算结束后隐藏进度条
            progress.pack_forget()
            status_label.config(text="")

    def confirm():
        error_label.config(text="")
        result_label.config(text="推荐结果将显示在这里...", foreground="black")
        status_label.config(text="")
        try:
            user_id = int(entry.get())
            threading.Thread(target=recommend_thread, args=(user_id,), daemon=True).start()
        except ValueError:
            error_label.config(text="请输入有效的ID！")

    confirm_btn = ttk.Button(root, text="获取推荐", command=confirm)
    confirm_btn.pack(pady=10)

    result_frame.pack(pady=10, fill="x", padx=20)

    root.mainloop()

get_user_id_and_recommend()