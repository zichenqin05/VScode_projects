import mainModel as mm
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

def get_user_id_and_recommend():
    root = tk.Tk()
    root.title("Short Video Recommendation System")
    root.geometry("500x400")

    large_font = ("SimHei", 16)
    result_font = ("SimHei", 12)

    label = ttk.Label(root, text="Please enter user ID:", font=large_font)
    label.pack(pady=20)

    entry = ttk.Entry(root, font=large_font, width=15)
    entry.pack()

    error_label = ttk.Label(root, text="", font=("SimHei", 12), foreground="red")
    error_label.pack(pady=10)

    status_label = ttk.Label(root, text="", font=("SimHei", 12), foreground="blue")
    status_label.pack(pady=5)

    progress = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate")
    # Hide progress bar by default
    progress.pack_forget()

    result_frame = ttk.Frame(root)
    result_label = ttk.Label(result_frame, text="Recommendation results will be shown here...", font=result_font, wraplength=350, justify="left")
    result_label.pack(pady=10)

    def recommend_thread(user_id):
        try:
            # Show progress bar and reset to 0
            progress.pack(pady=5)
            progress["value"] = 0
            status_label.config(text="Loading database...")
            root.update_idletasks()
            progress["value"] = 20
            time.sleep(0.5)

            status_label.config(text="Recommending high-quality videos based on the database and video quality...")
            root.update_idletasks()
            progress["value"] = 60

            recommendations = mm.main(user_id)

            status_label.config(text="Calculation completed!")
            progress["value"] = 100
            root.update_idletasks()
            time.sleep(0.5)  # Show full bar for a moment

            # Support dict or list
            if isinstance(recommendations, dict) and recommendations:
                result_text = f"Recommended videos for user {user_id}:\n"
                for tag, vid in recommendations.items():
                    result_text += f"Category {tag} recommended video ID: {vid}\n"
                result_label.config(text=result_text, foreground="black")
            elif isinstance(recommendations, list) and recommendations:
                result_text = f"Recommended videos for user {user_id}:\n" + "\n".join([f"- {item}" for item in recommendations[:5]])
                result_label.config(text=result_text, foreground="black")
            else:
                result_label.config(text="No recommendation found", foreground="orange")
        except Exception as e:
            result_label.config(text="No recommendation found", foreground="orange")
            messagebox.showerror("Error", f"Recommendation process error: {str(e)}")
        finally:
            # Hide progress bar after calculation
            progress.pack_forget()
            status_label.config(text="")

    def confirm():
        error_label.config(text="")
        result_label.config(text="Recommendation results will be shown here...", foreground="black")
        status_label.config(text="")
        try:
            user_id = int(entry.get())
            threading.Thread(target=recommend_thread, args=(user_id,), daemon=True).start()
        except ValueError:
            error_label.config(text="Please enter a valid ID!")

    confirm_btn = ttk.Button(root, text="Get Recommendation", command=confirm)
    confirm_btn.pack(pady=10)

    result_frame.pack(pady=10, fill="x", padx=20)

    root.mainloop()

get_user_id_and_recommend()