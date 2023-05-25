def send(message):
    # setup discord notification using discordwebhook
    import discordwebhook
    discord = discordwebhook.Discord(
        url='https://discord.com/api/webhooks/1100082805897187539/gshRvxkndkh9VIqr6C--wZip6LzhOhEUhvPzS6Jcddwp4Lj9KvzgZK4fHU8sO-NlmBIJ'
    )

    discord.post(
        content=message + '\n @everyone',
        username='ABSA Notification'
    )


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

