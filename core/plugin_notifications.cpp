#include "plugin_notifications.hpp"
#include "ui/ui_root.hpp"
#include "ui/console_notification_target.hpp"
#include "popup_ui/popup_ui.hpp"
void publishPluginNotification(const std::string& title, const std::string& message) {
    UIRoot::get().postTask([text = "[" + title + "] " + message] {
        auto& root = UIRoot::get();
        if (auto console = dynamic_cast<ConsoleNotificationTarget*>(root.getPanelByKey("console"))) {
            console->appendNotification(text);
            root.setVisible("console", true);
        }
        showPopup(); notifyPopupActivity();
    });
}
