#include "plugin_notifications.hpp"
#include "ui/ui_root.hpp"
#include "ui/console_panel.hpp"
#include "popup_ui/popup_ui.hpp"
void publishPluginNotification(const std::string& title, const std::string& message) {
    UIRoot::get().postTask([text = "[" + title + "] " + message] {
        auto& root = UIRoot::get();
        if (auto console = std::dynamic_pointer_cast<ConsolePanel>(root.getPanelByKey("console"))) {
            console->getHistory().push(text, 0xFF66CCFF, "system");
            root.setVisible("console", true);
        }
        showPopup(); notifyPopupActivity();
    });
}
