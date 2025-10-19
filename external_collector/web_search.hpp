#include pragma



class GRIMWS {
public:
    static void initialize();
    static void shutdown();
    static void StartSearch();
    static void SaveSearch();
    

    
    static bool isDown(MouseButton btn); 
    static bool wasPressed(MouseButton btn);
    static bool wasReleased(MouseButton btn);
    static void endFrame();
    static void onPress(MouseButton btn, std::function<void(MouseButton)> cb);
    static void onRelease(MouseButton btn, std::function<void(MouseButton)> cb);

private:
    static std::unordered_map<MouseButton, MouseState> buttonStates;
    static HHOOK mouseHook;
    static LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam);
    static void setDown(MouseButton btn);
    static void setUp(MouseButton btn);
};
