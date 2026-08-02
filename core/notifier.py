# core/notifier.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class QuantNotifier:
    def __init__(self, sender_email: str, sender_password: str, receiver_email: str):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
        # 以常见的 QQ邮箱 或 163邮箱 SMTP 为例
        self.smtp_server = "smtp.qq.com" 
        self.smtp_port = 465

    def send_trade_signal(self, ticker: str, action: str, probability: float):
        """
        当模型生成买卖信号时，发送邮件到手机端
        """
        subject = f"【A股量化警报】股票 {ticker} 产生交易信号：{action}"
        body = f"尊敬的量化交易员：\n\n机器学习模型已完成最新一轮扫描。\n- 标的代码: {ticker}\n- 预测上涨概率: {probability*100:.2f}%\n- 建议动作: {action}\n\n请及时登录实盘终端进行决策。"

        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        try:
            # 使用 SSL 安全连接发送邮件
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.receiver_email, message.as_string())
            server.quit()
            print(f"[{ticker}] 交易信号邮件已成功发送至手机！")
        except Exception as e:
            print(f"邮件发送失败: {e}")